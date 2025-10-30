"""
Main simulation functions for the election simulation.
"""
import os
import random
import pandas as pd
import numpy as np
import copy
import networkx as nx
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import re
import psutil
import time
from typing import Dict, List, Any, Tuple, Optional
import pickle
from pathlib import Path
import gc

# Import from other modules
from src.agents.agent import Agent
from src.agents.persona import load_profiles_and_network, save_all_personas
from src.models.feed_ranking import FeedRankingAlgorithm
from src.simulation.simulation_utils import save_population_summary, save_user_interactions, determine_voting_outcome
from src.utils.llm_utils import gen_completion

# Import configuration parameters
from src.config import (
    content_gen_prob, max_activity_interval,
    p_interest, p_followed, p_trending, p_random
)

# Memory monitoring functions
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def log_memory_usage(message=""):
    """Log current memory usage with an optional message."""
    mem_usage = get_memory_usage()
    print(f"Memory usage: {mem_usage:.2f} MB - {message}")

def is_memory_critical(threshold_mb=None):
    """Check if memory usage is approaching critical levels.
    
    If threshold_mb is None, dynamically calculate threshold as 70% of available memory.
    """
    # Get current memory usage
    mem_usage = get_memory_usage()
    
    # If no threshold provided, use 70% of available memory
    if threshold_mb is None:
        total_memory = psutil.virtual_memory().total / (1024 * 1024)  # Total memory in MB
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Available memory in MB
        
        # Use 70% of available memory as threshold
        dynamic_threshold = available_memory * 0.7
        
        # Ensure at least 1GB remains free
        min_free_memory = 1024  # 1GB in MB
        dynamic_threshold = min(dynamic_threshold, total_memory - min_free_memory)
        
        return mem_usage > dynamic_threshold
    else:
        # Use provided threshold
        return mem_usage > threshold_mb

def process_agent(agent_tuple, round_num, feed_length, local_contents_ref, local_agents_ref, election_day):
    """Process a single agent's engagement for a round."""
    agent_id, _ = agent_tuple
    
    # Get the agent from shared local_agents reference
    agent = local_agents_ref[agent_id]
    
    # Use local references to shared memory objects
    local_contents = local_contents_ref
    local_agents = local_agents_ref
        
    # Create lightweight state updates dict instead of copying the entire agent
    updates = {
        'engagement_history': {},
        'content_seen_rounds': {},
        'seen_content': set(),
        'likes': set(),
        'replies': {},
        'content_last_seen_replies': {},
        'conversation_history': None,
        'full_conversation_history': None,
        'next_activity_time': None,
        'following': None,  # Will store updated following list
        'follow_changes': [],  # Track follow/unfollow actions
        'voting_likelihood': None,  # Add this to capture voting likelihood
        'voting_history': {}  # Add this to capture voting history updates
    }
    
    # Determine candidate content
    candidate_contents = []
    for content in local_contents:
        content_id = content['content_id']
        # Check if agent has not seen the content before, or if it has new replies
        last_seen_replies = agent.content_last_seen_replies.get(content_id, -1)
        current_replies = len(content['replies'])
        if content_id not in agent.seen_content or current_replies > last_seen_replies:
            # Add author's public profile to content for display
            if 'author_profile' not in content:
                author_id = content['author_id']
                if author_id in local_agents:
                    content['author_profile'] = local_agents[author_id].persona['public_profile']
                else:
                    content['author_profile'] = "Unknown"
            candidate_contents.append(content)

    if not candidate_contents:
        # Even if there are no contents to engage with, we should preserve the conversation history
        updates['conversation_history'] = agent.conversation_history
        updates['full_conversation_history'] = agent.full_conversation_history
        updates['next_activity_time'] = agent.next_activity_time
        return agent_id, 0, [], updates, None  # Return updates with no changes

    # Contents matching agent's interests
    user_interests = agent.persona['interests'].split(',') if ',' in agent.persona['interests'] else [agent.persona['interests']]
    interest_contents = [content for content in candidate_contents if content['subtopic'] in user_interests]

    # Contents from authors the agent follows
    followed_author_contents = [content for content in candidate_contents if content['author_id'] in agent.following]

    # Trending contents (top by total engagement)
    trending_contents = sorted(candidate_contents, key=lambda x: x['total_engagement'], reverse=True)
    trending_contents = [content for content in trending_contents if content not in interest_contents]

    # Random contents
    random_contents = [content for content in candidate_contents if content not in interest_contents and content not in trending_contents]

    feed_length_actual = min(feed_length, len(candidate_contents))
    n_interest = int(feed_length_actual * p_interest)
    n_followed = int(feed_length_actual * p_followed)
    n_trending = int(feed_length_actual * p_trending)
    n_random = feed_length_actual - n_interest - n_followed - n_trending

    # Sample contents
    interest_sample = random.sample(interest_contents, min(n_interest, len(interest_contents)))
    followed_sample = random.sample(followed_author_contents, min(n_followed, len(followed_author_contents)))
    trending_sample = trending_contents[:n_trending]
    random_sample = random.sample(random_contents, min(n_random, len(random_contents)))

    # Combine samples
    feed_contents = interest_sample + followed_sample + trending_sample + random_sample

    # Fill up feed if necessary
    if len(feed_contents) < feed_length_actual:
        remaining_contents = [content for content in candidate_contents if content not in feed_contents]
        additional_contents = random.sample(remaining_contents, min(feed_length_actual - len(feed_contents), len(remaining_contents)))
        feed_contents.extend(additional_contents)

    # Apply feed ranking algorithm based on treatment
    ranking_algorithm = FeedRankingAlgorithm()

    # Rank feed content
    ranked_contents = ranking_algorithm.rank(agent, feed_contents, round_num)

    # Copy existing state from agent before simulation
    updates['seen_content'] = agent.seen_content.copy()
    updates['likes'] = agent.likes.copy()
    updates['replies'] = agent.replies.copy()
    updates['content_last_seen_replies'] = agent.content_last_seen_replies.copy()

    # Simulate engagement decisions - this will update agent.conversation_history directly
    new_content = None
    try:
        decisions = agent.decide_engagement(
            ranked_contents, 
            round_num, 
            election_day  # This is max(time_points) from run_simulation
        )
        
        # Extract engagement decisions, generated content, and voting likelihood
        engagement_decisions = decisions.get("engagement", [])
        generated_content_text = decisions.get("generated_content", "")
        voting_likelihood = decisions.get("voting_likelihood", 0)
        
        # Store the voting likelihood in the updates dictionary
        updates['voting_likelihood'] = voting_likelihood
        updates['voting_history'][round_num] = voting_likelihood
        
        # Create new content if generated and not empty string (agent decided to post)
        if generated_content_text and len(generated_content_text.strip()) > 0:
            # Extract topic and subtopic from interests if available
            content_topic = "general"
            content_subtopic = random.choice(user_interests) if user_interests else "general"
            
            # Don't assign content_id yet - this will be done centrally after processing
            new_content = {
                'author_id': agent.user_id,
                'author_profile': agent.persona['public_profile'],
                'text': generated_content_text,
                'replies': [],
                'likes': 0,
                'topic': content_topic,
                'subtopic': content_subtopic,
                'timestamp': round_num,
                'total_engagement': 0,
                'engaged_users': set()
            }
    except Exception as e:
        print(f"Error in decide_engagement for agent {agent_id}: {e}")
        engagement_decisions = []  # Default to no engagement on error
        voting_likelihood = random.randint(0, 4)  # Default voting likelihood on error
        agent.voting_likelihood = voting_likelihood
        agent.voting_history[round_num] = voting_likelihood
    
    # IMPORTANT: After decide_engagement is called, get the updated conversation histories
    # The decide_engagement method modifies these histories directly on the agent
    updates['conversation_history'] = agent.conversation_history
    updates['full_conversation_history'] = agent.full_conversation_history
    updates['next_activity_time'] = agent.next_activity_time
    
    # Make a copy of the current following list
    updates['following'] = agent.following.copy()

    # Track feed content for this round
    updates['engagement_history'][round_num] = [content['content_id'] for content in ranked_contents]
    
    # Track all rounds when content is seen
    for content in ranked_contents:
        content_id = content['content_id']
        # Copy existing content_seen_rounds for this content_id
        if content_id in agent.content_seen_rounds:
            updates['content_seen_rounds'][content_id] = agent.content_seen_rounds[content_id].copy()
        else:
            updates['content_seen_rounds'][content_id] = []
            
        if round_num not in updates['content_seen_rounds'][content_id]:
            updates['content_seen_rounds'][content_id].append(round_num)

    engaged = 0
    engagement_results = []
    
    # Process engagement decisions
    if not isinstance(engagement_decisions, list):
        print(f"Warning: engagement_decisions is not a list for agent {agent_id}, type: {type(engagement_decisions)}")
        engagement_decisions = []  # Set to empty list
        
    for decision in engagement_decisions:
        try:
            # Ensure decision is a dictionary
            if not isinstance(decision, dict):
                print(f"Skipping non-dictionary decision for agent {agent_id}: {decision}")
                continue
                
            # Get values with defaults if missing
            post_number = decision.get('post_number', 0)
            engage = decision.get('engage', 'nothing').lower()
            reply_text = decision.get('reply_text', '')
            follow_action = decision.get('follow_action', 'no_change').lower()
            
            # Validate values
            if not isinstance(post_number, int) or post_number < 1 or post_number > len(ranked_contents):
                continue
                
            if reply_text is not None:
                reply_text = reply_text.strip()
            else:
                reply_text = ''

            content = ranked_contents[post_number - 1]
            content_id = content['content_id']
            author_id = content['author_id']

            # Track that the user has seen this content
            updates['seen_content'].add(content_id)

            # Handle engagement actions (like/reply)
            if engage == 'like':
                if content_id not in updates['likes']:
                    # Verify content exists in local_contents before adding to likes
                    if any(c['content_id'] == content_id for c in local_contents):
                        updates['likes'].add(content_id)
                        engagement_results.append(('like', content_id))
                        engaged += 1
                    else:
                        print(f"Warning: Agent {agent_id} tried to like non-existent content ID {content_id}")
            elif engage == 'reply':
                if content_id not in updates['replies']:
                    # Verify content exists in local_contents before adding reply
                    if any(c['content_id'] == content_id for c in local_contents):
                        reply_text = reply_text[:140]
                        updates['replies'][content_id] = reply_text
                        engagement_results.append(('reply', content_id, reply_text))
                        engaged += 1
                    else:
                        print(f"Warning: Agent {agent_id} tried to reply to non-existent content ID {content_id}")

            # Handle follow/unfollow actions
            # Don't follow/unfollow yourself
            if author_id != agent_id:
                if follow_action == 'follow' and author_id not in updates['following']:
                    updates['following'].append(author_id)
                    updates['follow_changes'].append(('follow', author_id))
                elif follow_action == 'unfollow' and author_id in updates['following']:
                    updates['following'].remove(author_id)
                    updates['follow_changes'].append(('unfollow', author_id))

            # Update agent's last seen replies count
            updates['content_last_seen_replies'][content_id] = len(content['replies'])
        except Exception as e:
            print(f"Error processing engagement decision for agent {agent_id}: {e}")
            continue
    
    # Count content generation as engagement if it happened
    if new_content is not None:
        engaged += 1

    return agent_id, engaged, engagement_results, updates, new_content

# Define process_voting function outside run_simulation to make it picklable
def process_voting_with_day(agent_tuple_with_day):
    """Process voting for a single agent with the election day passed as part of the tuple."""
    agent_id, agent, day = agent_tuple_with_day
    voted = determine_voting_outcome(agent, day)
    # Also return the updated conversation history
    return agent_id, voted, agent.full_conversation_history

def run_simulation(n_users, time_points, treatment_probs, topic, c, feed_length, election_day, random_seed=None, treatment_seed=None, is_warmup=False, initial_state=None, output_dir=None, start_round=1, n_cores=None, setting_name=None, batch_size=1000):
    """
    Run a single simulation with optional initial state from warmup period.
    
    Args:
        n_users: Number of users in simulation
        time_points: List of time points for the simulation
        treatment_probs: List of treatment probabilities for each time point
        topic: Topic for content generation
        c: Initial number of content pieces
        feed_length: Length of feed for each user
        random_seed: Random seed for reproducibility (all random processes except treatment)
        treatment_seed: Random seed for treatment allocation only
        is_warmup: Boolean indicating if this is a warmup period
        initial_state: Dictionary containing warmup period final state (agents, contents, mapped_G)
        output_dir: Directory to save output files
        start_round: Starting round number (default 1, negative for warmup)
        n_cores: Number of CPU cores to use for parallel processing. If None, uses all available cores.
        election_day: The day of the election (default: max(time_points) if None)
        setting_name: Name of the treatment setting (e.g., "treatment_info_message")
        batch_size: Number of agents to process in one batch (default: 1000)
    
    Returns:
        tuple: (engagement_data, treatment_data, activity_data, voting_data, final_state)
        where final_state is a dict containing:
        {
            'agents': dict of agents,
            'contents': list of contents,
            'mapped_G': networkx DiGraph
        }
    """
    # Start memory monitoring
    log_memory_usage("Starting simulation")
    
    # Set pandas option to use future behavior
    pd.set_option('future.no_silent_downcasting', True)
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # If treatment_seed is None, use random_seed for treatment assignment
    if treatment_seed is None:
        treatment_seed = random_seed

    if initial_state is not None:
        # Validate state
        assert all(k in initial_state for k in ['agents', 'contents', 'mapped_G']), "Missing required state components"
        assert len(initial_state['agents']) == n_users, "Mismatched number of agents"
        
    # Calculate total rounds before checkpoint logic
    total_rounds = max(time_points)
    
    # Initialize treatment schedule
    treatment_schedule = {}  # round_number: set of user_ids in treatment

    # Check if a checkpoint exists and if we should resume from it
    checkpoint_dir = None
    if output_dir:
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Find the latest checkpoint if any
        checkpoints = list(checkpoint_dir.glob("checkpoint_round_*.pkl"))
        latest_checkpoint = None
        latest_round = start_round - 1
        
        for cp in checkpoints:
            match = re.search(r'checkpoint_round_(-?\d+)\.pkl', str(cp))
            if match:
                round_num = int(match.group(1))
                if round_num > latest_round:
                    latest_round = round_num
                    latest_checkpoint = cp
        
        # If checkpoint exists and is for a round >= start_round, load it
        if latest_checkpoint and latest_round >= start_round:
            print(f"Resuming from checkpoint at round {latest_round}")
            try:
                with open(latest_checkpoint, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Restore state from checkpoint
                local_agents = checkpoint_data['agents']
                local_contents = checkpoint_data['contents']
                local_mapped_G = checkpoint_data['mapped_G']
                
                # Also restore treatment schedule if available
                if 'treatment_schedule' in checkpoint_data:
                    treatment_schedule = checkpoint_data['treatment_schedule']
                
                # Update start_round to continue from the checkpoint
                start_round = latest_round + 1
                print(f"Successfully loaded checkpoint. Continuing from round {start_round}")
                
                # Skip initialization since we loaded from checkpoint
                checkpoint_loaded = True
                checkpoint_data_frames_loaded = False  # Default to False, will set to True if loaded
                
                # Restore DataFrame data if available
                if all(k in checkpoint_data for k in ['engagement_data', 'treatment_data', 'activity_data', 'voting_data']):
                    engagement_data = checkpoint_data['engagement_data']
                    treatment_data = checkpoint_data['treatment_data']
                    activity_data = checkpoint_data['activity_data']
                    voting_data = checkpoint_data['voting_data']
                    print("Successfully loaded DataFrames from checkpoint")
                    checkpoint_data_frames_loaded = True
                else:
                    print("DataFrames not found in checkpoint, will initialize new ones")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch instead")
                checkpoint_loaded = False
                checkpoint_data_frames_loaded = False
        else:
            checkpoint_loaded = False
            checkpoint_data_frames_loaded = False
    else:
        checkpoint_loaded = False
        checkpoint_data_frames_loaded = False
    
    # Initialize simulation state if not loaded from checkpoint
    if not checkpoint_loaded:
        if initial_state is None:
            # Load personas and network from CSV files
            personas, network_df = load_profiles_and_network(n_users, random_seed)
            
            # Create agents with their personas
            local_agents = {}
            for user_id, persona in personas.items():
                local_agents[user_id] = Agent(user_id=user_id, persona=persona)
                # Set initial random activity time for each agent
                local_agents[user_id].next_activity_time = start_round + random.randint(1, max_activity_interval)
            
            # Create a directed graph from the network data
            G = nx.DiGraph()
            G.add_edges_from(zip(network_df['source_user_id'], network_df['target_user_id']))
            
            # Use the graph as is since we're already using the correct user IDs
            local_mapped_G = G
            
            # Assign following/followers based on the network
            for node in local_mapped_G.nodes():
                if node in local_agents:  # Only process nodes that exist in our agents
                    local_agents[node].following = list(local_mapped_G.successors(node))  # People this node follows
                    for neighbor in local_mapped_G.successors(node):
                        if neighbor in local_agents:  # Only process neighbors that exist in our agents
                            local_agents[neighbor].followers.append(node)  # Add this node as a follower

            # Generate initial content using LLM
            local_contents = []
            for content_id in range(c):
                # Randomly select a user to create this content
                author_id = random.choice(list(local_agents.keys()))
                author = local_agents[author_id]
                
                # Select a subtopic from the author's interests
                user_interests = author.persona['interests'].split(',') if ',' in author.persona['interests'] else [author.persona['interests']]
                subtopic = random.choice(user_interests)
                
                prompt = f"Generate a social media post, rules:max140chars;topic:{topic};subtopic:{subtopic}\nPost:"
                # Use the author's model to generate content
                response = gen_completion([{'role': 'user', 'content': prompt}], model=author.model)
                content_text = response.strip()[:140]  # Limit content to 140 characters
                content = {
                    'content_id': content_id,
                    'author_id': author_id,  # Set the author_id to the user who created it
                    'author_profile': author.persona['public_profile'],  # Include author's public profile
                    'text': content_text,
                    'replies': [],
                    'likes': 0,
                    'topic': topic,
                    'subtopic': subtopic,
                    'timestamp': 0,
                    'total_engagement': 0,
                    'engaged_users': set()
                }
                local_contents.append(content)

            # Save population summary and all personas
            if output_dir:
                save_population_summary(local_agents, os.path.join(output_dir, 'population_summary.txt'))
                save_all_personas(local_agents, os.path.join(output_dir, 'personas.csv'))
            else:
                save_population_summary(local_agents, 'population_summary.txt')
                save_all_personas(local_agents, 'personas.csv')
        else:
            # Use the provided initial state
            print("Loading initial state...")
            local_agents = copy.deepcopy(initial_state['agents'])
            local_contents = copy.deepcopy(initial_state['contents'])
            local_mapped_G = copy.deepcopy(initial_state['mapped_G'])
            log_memory_usage("After loading initial state")

        # Assign treatments over time periods if not loaded from checkpoint
        if not checkpoint_loaded or 'treatment_schedule' not in checkpoint_data:
            for idx, t in enumerate(time_points):
                pi = treatment_probs[idx]
                start_round_idx = time_points[idx-1] + 1 if idx > 0 else start_round
                end_round = t

                if pi == 0:
                    # All users are in control
                    treatment_users = set()
                elif pi == 1:
                    # All users are treated
                    treatment_users = set(local_agents.keys())
                else:
                    # Use binomial distribution to randomly assign treatment with probability pi
                    # Create a separate Random object with treatment_seed for treatment assignment only
                    treatment_rng = random.Random(treatment_seed)
                    treatment_users = {i for i in local_agents.keys() if treatment_rng.random() < pi}

                for round_num in range(start_round_idx, end_round+1):
                    if round_num not in treatment_schedule:
                        treatment_schedule[round_num] = set()
                    treatment_schedule[round_num].update(treatment_users)

        # Initialize DataFrames if not loaded from checkpoint
        if not checkpoint_loaded or not checkpoint_data_frames_loaded:
            # Panel data to collect engagement metrics - initialize with float type
            engagement_data = pd.DataFrame(0.0, index=local_agents.keys(), columns=range(start_round, total_rounds + 1))
            # Treatment data to record treatments (initialize to 0)
            treatment_data = pd.DataFrame(0, index=local_agents.keys(), columns=range(start_round, total_rounds + 1))
            # Activity data to record user activity (initialize to 0)
            activity_data = pd.DataFrame(0, index=local_agents.keys(), columns=range(start_round, total_rounds + 1))
            # Voting likelihood data (initialize to NaN)
            voting_data = pd.DataFrame(np.nan, index=local_agents.keys(), columns=range(start_round, total_rounds + 1))
            
            print("Initialized new DataFrames")
        
        log_memory_usage("After initialization")

    # Begin simulation
    for round_num in range(start_round, total_rounds + 1):
        round_start_time = time.time()
        
        if is_warmup:
            print(f"Warmup round {round_num}")
        else:
            print(f"Starting round {round_num}")
        
        log_memory_usage(f"Start of round {round_num}")
        
        # Collect likely voter data before processing agents
        likely_voters = []
        likely_voter_count = 0
        
        for agent_id, agent in local_agents.items():
            # Only consider users 18 and older for voting
            if agent.persona['age'] >= 18 and agent.voting_likelihood is not None and agent.voting_likelihood >= 3:
                likely_voters.append(agent_id)
                likely_voter_count += 1
        
        # Create a likely voter data object to share
        likely_voter_data = {
            'count': likely_voter_count,
            'users': likely_voters
        }
            
        # Provide each agent with necessary treatment information
        for agent_id, agent in local_agents.items():
            # Set treatment type based on setting_name
            if setting_name:
                agent.treatment_type = setting_name
            
            # Update likely voter data for social treatments
            if agent.persona['age'] >= 18 and (setting_name == "treatment_soc_message" or setting_name == "experiment_soc_message" or setting_name == "control_then_soc"):
                agent.likely_voter_data = likely_voter_data
            
        # Update treatments
        for agent_id in local_agents:
            # Only assign treatment to users 18 and older
            if local_agents[agent_id].persona['age'] >= 18:
                is_treated = agent_id in treatment_schedule.get(round_num, set())
                local_agents[agent_id].treatment = 1 if is_treated else 0
                treatment_data.at[agent_id, round_num] = local_agents[agent_id].treatment
            else:
                # Make sure users under 18 are not treated
                local_agents[agent_id].treatment = 0
                treatment_data.at[agent_id, round_num] = 0

        # Update total engagement counts
        for content in local_contents:
            content['total_engagement'] = content['likes'] + len(content['replies'])

        # For each agent, collect friend engagement data
        for agent in local_agents.values():
            agent.friend_like_counts = defaultdict(int)
            agent.friend_reply_counts = defaultdict(int)
            for friend_id in agent.following:
                friend = local_agents[friend_id]
                for content_id in friend.likes:
                    agent.friend_like_counts[content_id] += 1
                for content_id in friend.replies.keys():
                    agent.friend_reply_counts[content_id] += 1

        # Determine active users for this round
        active_users = []
        for agent_id, agent in local_agents.items():
            if agent.next_activity_time == round_num:
                active_users.append((agent_id, agent))
                # Mark user as active in this round
                activity_data.at[agent_id, round_num] = 1
                
        log_memory_usage(f"Round {round_num}: {len(active_users)} active users identified")

        # Process agents in parallel in batches
        if n_cores is None:
            n_cores = min(cpu_count(), 100)  # Cap at 100 cores to prevent overhead
        n_processes = min(n_cores, len(active_users))
        
        # Initialize container for all results
        all_results = []
        
        # Calculate an initial batch size based on available memory
        current_memory = get_memory_usage()
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # in MB
        
        # Use at most 50% of available memory per batch
        memory_per_user = max(1, current_memory / max(1, n_users))  # Avoid division by zero
        max_batch_size_by_memory = int(available_memory * 0.5 / memory_per_user)
        
        # Use the smaller of user-specified batch size and memory-based batch size
        adaptive_batch_size = min(batch_size, max_batch_size_by_memory)
        adaptive_batch_size = max(50, adaptive_batch_size)  # Ensure minimum of 50 users per batch
        
        print(f"Using adaptive batch size of {adaptive_batch_size} users (from initial {batch_size})")
        
        # Process in batches to manage memory
        for batch_start in range(0, len(active_users), adaptive_batch_size):
            # Check if memory is critical before processing next batch
            if is_memory_critical():
                print("WARNING: Memory usage is critical. Attempting to free memory...")
                # Force garbage collection
                gc.collect()
                
                # If still critical after GC, reduce batch size
                if is_memory_critical():
                    remaining_batches = (len(active_users) - batch_start) // adaptive_batch_size + 1
                    if remaining_batches > 1 and adaptive_batch_size > 400:
                        old_batch_size = adaptive_batch_size
                        adaptive_batch_size = max(400, adaptive_batch_size // 2)
                        print(f"Reducing batch size from {old_batch_size} to {adaptive_batch_size} for remaining batches")
            
            batch_end = min(batch_start + adaptive_batch_size, len(active_users))
            batch = active_users[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//adaptive_batch_size + 1}/{(len(active_users)-1)//adaptive_batch_size + 1} with {len(batch)} users")
            
            if n_processes > 1 and len(batch) > 1:
                try:
                    # Create a multiprocessing Manager to handle shared objects
                    with Manager() as manager:
                        # Create shared references to agents and contents
                        shared_agents = manager.dict(local_agents)
                        shared_contents = manager.list(local_contents)
                        
                        with Pool(min(n_processes, len(batch))) as pool:
                            process_func = partial(process_agent, 
                                                round_num=round_num,
                                                feed_length=feed_length,
                                                local_contents_ref=shared_contents,
                                                local_agents_ref=shared_agents,
                                                election_day=election_day)
                            # Replace map with imap_unordered and process results as they arrive
                            batch_results = []
                            for result in pool.imap_unordered(process_func, batch):
                                batch_results.append(result)
                except Exception as e:
                    print(f"Error in pool processing: {e}")
                    # Fallback to non-parallel processing if pool fails
                    batch_results = [process_agent(agent_tuple, round_num, feed_length, local_contents, local_agents, election_day) 
                                   for agent_tuple in batch]
            else:
                batch_results = [process_agent(agent_tuple, round_num, feed_length, local_contents, local_agents, election_day) 
                               for agent_tuple in batch]
            
            all_results.extend(batch_results)
            log_memory_usage(f"After batch {batch_start//adaptive_batch_size + 1}")

        # Process results and update engagement
        new_content_items = []  # Collect all new content generated this round
        for agent_id, engaged, engagement_results, updates, new_content in all_results:
            # Update the agent's state with the processed updates
            if updates:
                agent = local_agents[agent_id]
                
                # Update engagement_history
                for round_key, content_ids in updates['engagement_history'].items():
                    agent.engagement_history[round_key] = content_ids
                
                # Update content_seen_rounds
                for content_id, rounds_list in updates['content_seen_rounds'].items():
                    agent.content_seen_rounds[content_id] = rounds_list
                
                # Update sets and dictionaries
                agent.seen_content = updates['seen_content']
                agent.likes = updates['likes']
                agent.replies = updates['replies']
                agent.content_last_seen_replies = updates['content_last_seen_replies']
                
                # Update conversation history
                agent.conversation_history = updates['conversation_history']
                agent.full_conversation_history = updates['full_conversation_history']
                
                # Update next activity time
                agent.next_activity_time = updates['next_activity_time']
                
                # Update following list if changed
                if updates['following'] is not None:
                    follow_changes = updates.get('follow_changes', [])
                    
                    # Process follow/unfollow actions
                    for action, target_id in follow_changes:
                        if action == 'follow':
                            # Add this agent as a follower to the target
                            if agent_id not in local_agents[target_id].followers:
                                local_agents[target_id].followers.append(agent_id)
                        elif action == 'unfollow':
                            # Remove this agent as a follower from the target
                            if agent_id in local_agents[target_id].followers:
                                local_agents[target_id].followers.remove(agent_id)
                    
                    # Update the agent's following list
                    agent.following = updates['following']
            
            # Collect new content if generated
            if new_content is not None:
                new_content_items.append((agent_id, new_content))
            
            engagement_data.at[agent_id, round_num] = engaged
            
            # Update voting data - only for users 18 and older
            if updates['voting_likelihood'] is not None and agent.persona['age'] >= 18:
                agent.voting_likelihood = updates['voting_likelihood']
                voting_data.at[agent_id, round_num] = agent.voting_likelihood
                
            # Update voting history - only for users 18 and older
            if agent.persona['age'] >= 18:
                for round_key, likelihood in updates['voting_history'].items():
                    agent.voting_history[round_key] = likelihood
            
            # Update content engagement
            for result in engagement_results:
                if result[0] == 'like':
                    content_id = result[1]
                    # Find the content or skip if not found
                    try:
                        content = next(c for c in local_contents if c['content_id'] == content_id)
                        content['likes'] += 1
                        content['engaged_users'].add(agent_id)
                    except StopIteration:
                        print(f"Warning: Content ID {content_id} not found for like action by agent {agent_id}")
                        continue
                elif result[0] == 'reply':
                    content_id = result[1]
                    reply_text = result[2]
                    # Find the content or skip if not found
                    try:
                        content = next(c for c in local_contents if c['content_id'] == content_id)
                        content['replies'].append({
                            'user_id': agent_id,
                            'text': reply_text
                        })
                        content['engaged_users'].add(agent_id)
                    except StopIteration:
                        print(f"Warning: Content ID {content_id} not found for reply action by agent {agent_id}")
                        continue
        
        # Process all new content generated this round
        if new_content_items:
            for _, item in enumerate(new_content_items):
                agent_id, content = item
                content_id = len(local_contents)
                content['content_id'] = content_id
                
                # Add author's public profile to the content
                if 'author_profile' not in content and agent_id in local_agents:
                    content['author_profile'] = local_agents[agent_id].persona['public_profile']
                
                local_contents.append(content)

        # After processing results
        log_memory_usage(f"After processing results for round {round_num}")
            
        # Check for memory issues again after adding new content
        if is_memory_critical():
            print("WARNING: Memory is critical after processing round. Attempting cleanup...")
            gc.collect()

        # Reset multiprocessing pool to free resources at the end of each round
        if 'pool' in locals() and hasattr(pool, 'close'):
            pool.close()
            pool.join()
            del pool
            gc.collect()
            print(f"Multiprocessing pool reset after round {round_num}")

        print(f"Round {round_num} completed.")
        round_engagements = engagement_data[round_num].fillna(0)
        avg_engagement = round_engagements.mean()
        print(f"Average engagement for round {round_num}: {avg_engagement:.2f}")
        print(f"Round {round_num}: {len(active_users)} active users out of {len(local_agents)} total users ({len(active_users)/len(local_agents)*100:.1f}%)")
        
        # Calculate and print average voting likelihood for active users in this round
        active_eligible_users = [agent_id for agent_id, agent in active_users if agent.persona['age'] >= 18]
        if active_eligible_users:
            active_voting_likelihoods = [local_agents[agent_id].voting_likelihood for agent_id in active_eligible_users if local_agents[agent_id].voting_likelihood is not None]
            if active_voting_likelihoods:
                avg_voting_likelihood = sum(active_voting_likelihoods) / len(active_voting_likelihoods)
                print(f"Round {round_num}: Average voting likelihood for active eligible voters: {avg_voting_likelihood:.2f}/4 (from {len(active_voting_likelihoods)} users)")
        
        # Log round timing and memory
        round_time = time.time() - round_start_time
        print(f"Round {round_num} took {round_time:.2f} seconds")
        log_memory_usage(f"End of round {round_num}")

        # After round is complete, save checkpoint
        if checkpoint_dir:
            try:
                checkpoint_file = checkpoint_dir / f"checkpoint_round_{round_num}.pkl"
                print(f"Saving checkpoint for round {round_num} to {checkpoint_file}")
                
                # Create checkpoint data
                checkpoint_data = {
                    'agents': local_agents,
                    'contents': local_contents,
                    'mapped_G': local_mapped_G,
                    'engagement_data': engagement_data,
                    'treatment_data': treatment_data,
                    'activity_data': activity_data,
                    'voting_data': voting_data,
                    'round_num': round_num,
                    'treatment_schedule': treatment_schedule  # Also save treatment schedule
                }
                
                # Save checkpoint
                checkpoint_start_time = time.time()
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"Checkpoint saved in {time.time() - checkpoint_start_time:.2f} seconds")
                
                # Remove older checkpoints, keeping only the last 2
                checkpoints = list(checkpoint_dir.glob("checkpoint_round_*.pkl"))
                # Sort checkpoints by round number, not filename string
                checkpoint_rounds = []
                for cp in checkpoints:
                    match = re.search(r'checkpoint_round_(-?\d+)\.pkl', str(cp))
                    if match:
                        round_num_in_filename = int(match.group(1))
                        checkpoint_rounds.append((round_num_in_filename, cp))

                # Sort by round number and keep only the last 2
                checkpoint_rounds.sort(key=lambda x: x[0])
                checkpoints_to_delete = [cp for _, cp in checkpoint_rounds[:-2]]

                for old_cp in checkpoints_to_delete:  # Keep last 2 checkpoints
                    try:
                        os.remove(old_cp)
                    except:
                        pass
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    if is_warmup:
        print("Warmup period completed.")
    else:
        print("Simulation completed.")
        
        # Create and save pre-voting state
        # Ensure all agents have valid conversation history before returning
        for agent_id, agent in local_agents.items():
            if agent.conversation_history is None:
                agent.conversation_history = []
            if agent.full_conversation_history is None:
                agent.full_conversation_history = []
                
        # Create pre-voting state
        pre_voting_state = {
            'agents': local_agents,  
            'contents': local_contents,  
            'mapped_G': local_mapped_G  
        }
        
        # Save pre-voting state if output_dir is provided
        if output_dir:
            pre_voting_dir = Path(output_dir) / "pre_voting"
            pre_voting_dir.mkdir(exist_ok=True)
            pre_voting_state_file = pre_voting_dir / "pre_voting_state.pkl"
            print(f"Saving pre-voting state to {pre_voting_state_file}")
            with open(pre_voting_state_file, 'wb') as f:
                pickle.dump(pre_voting_state, f)
        
        # After simulation is complete, determine voting outcomes for all agents
        print("Determining voting outcomes...")
        
        # Ensure n_cores and n_processes are defined (in case we resumed from a checkpoint after all rounds)
        if n_cores is None:
            n_cores = min(cpu_count(), 100)

        # Process voting in parallel using the same number of cores
        eligible_agents = [(agent_id, agent, election_day) for agent_id, agent in local_agents.items() if agent.voted is None]
        n_processes = min(n_cores, len(eligible_agents))
        
        # Process in batches to manage memory
        for batch_start in range(0, len(eligible_agents), batch_size):
            batch_end = min(batch_start + batch_size, len(eligible_agents))
            batch = eligible_agents[batch_start:batch_end]
            
            print(f"Processing voting batch {batch_start//batch_size + 1}/{(len(eligible_agents)-1)//batch_size + 1} with {len(batch)} agents")
            
            if n_processes > 1 and len(batch) > 1:
                try:
                    with Pool(min(n_processes, len(batch))) as pool:
                        # Replace map with imap_unordered for voting processing too
                        voting_results = []
                        for result in pool.imap_unordered(process_voting_with_day, batch):
                            voting_results.append(result)
                except Exception as e:
                    print(f"Error in voting pool processing: {e}")
                    # Fallback to non-parallel processing if pool fails
                    voting_results = [process_voting_with_day(agent_tuple) for agent_tuple in batch]
            else:
                voting_results = [process_voting_with_day(agent_tuple) for agent_tuple in batch]
            
            # Update agent voted status and conversation history
            for result in voting_results:
                agent_id, voted, conversation_history = result
                local_agents[agent_id].voted = voted
                # Update the conversation history from the worker process
                local_agents[agent_id].full_conversation_history = conversation_history
        
        print("Voting outcomes determined.")
        
    # Return the final state along with the data
    # Ensure all agents have valid conversation history before returning
    for agent_id, agent in local_agents.items():
        if agent.conversation_history is None:
            agent.conversation_history = []
        if agent.full_conversation_history is None:
            agent.full_conversation_history = []
            
    # Create final state
    final_state = {
        'agents': local_agents,  
        'contents': local_contents,  
        'mapped_G': local_mapped_G  
    }
    
    log_memory_usage("After creating final state")
    
    # Filter out users under 18 from voting_data and treatment_data
    eligible_voter_ids = [agent_id for agent_id, agent in local_agents.items() if agent.persona['age'] >= 18]
    voting_data_filtered = voting_data.loc[eligible_voter_ids]
    treatment_data_filtered = treatment_data.loc[eligible_voter_ids]
    
    return engagement_data, treatment_data_filtered, activity_data, voting_data_filtered, final_state 