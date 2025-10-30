"""
Utility functions for the simulation.
"""
import os
from collections import defaultdict
import pandas as pd
import random
import re
import csv
import networkx as nx

from src.utils.llm_utils import gen_completion
from src.agents.agent import assign_persona_to_model

def save_population_summary(agents, output_file):
    """
    Save a summary of the population demographics, interests, network connectivity, and model distribution to a file.
    
    Args:
        agents: Dictionary of all agents
        output_file: Path to save the summary
    """
    with open(output_file, 'w') as f:
        f.write("\n=== Population Summary ===\n")
        
        # Demographic statistics
        gender_counts = defaultdict(int)
        race_counts = defaultdict(int)
        age_groups = defaultdict(int)
        interest_counts = defaultdict(int)
        model_counts = defaultdict(int)
        following_counts = []
        
        for agent in agents.values():
            # Count demographics
            gender_counts[agent.persona['gender']] += 1
            race_counts[agent.persona['race/ethnicity']] += 1
            age = agent.persona['age']
            age_group = f"{age//10*10}-{age//10*10+9}" if age < 90 else "90+"
            age_groups[age_group] += 1
            
            # Count interests
            for interest in agent.persona['interests'].split(','):
                interest_counts[interest.strip()] += 1
                
            # Count models
            model_counts[agent.model] += 1
                
            # Track network connectivity
            following_counts.append(len(agent.following))
        
        # Write demographic summary
        f.write("\nDemographics:\n")
        f.write(f"Total population: {len(agents)}\n")
        
        f.write("\nGender distribution:\n")
        for gender, count in gender_counts.items():
            f.write(f"{gender}: {count} ({count/len(agents)*100:.1f}%)\n")
        
        f.write("\nRace/Ethnicity distribution:\n")
        for race, count in race_counts.items():
            f.write(f"{race}: {count} ({count/len(agents)*100:.1f}%)\n")
        
        f.write("\nAge distribution:\n")
        for age_group, count in sorted(age_groups.items()):
            f.write(f"{age_group}: {count} ({count/len(agents)*100:.1f}%)\n")
        
        # Write interests summary
        f.write("\nInterest distribution:\n")
        total_interests = sum(interest_counts.values())
        for interest, count in sorted(interest_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{interest}: {count} ({count/total_interests*100:.1f}%)\n")
        
        # Write model distribution
        f.write("\nModel distribution:\n")
        for model, count in model_counts.items():
            f.write(f"{model}: {count} ({count/len(agents)*100:.1f}%)\n")
        
        # Write network connectivity summary
        f.write("\nNetwork connectivity:\n")
        f.write(f"Average number of connections: {sum(following_counts)/len(following_counts):.1f}\n")
        f.write(f"Minimum connections: {min(following_counts) if following_counts else 0}\n")
        f.write(f"Maximum connections: {max(following_counts) if following_counts else 0}\n")
        f.write("="*50 + "\n")

def save_user_interactions(user_id, agents, contents, output_file):
    """
    Save detailed information about a user's interactions, including their engagement history,
    the content they've seen, their social network, and their voting history.
    
    Args:
        user_id: The ID of the user to analyze
        agents: Dictionary of all agents
        contents: List of all content items
        output_file: Path to save the detailed analysis
    """
    if user_id not in agents:
        print(f"User {user_id} not found in agents!")
        return
    
    user = agents[user_id]
    
    with open(output_file, 'w') as f:
        f.write(f"\n=== User {user_id} Analysis ===\n\n")
        
        # User demographics
        f.write("User Demographics:\n")
        f.write(f"Age: {user.persona['age']}\n")
        f.write(f"Gender: {user.persona['gender']}\n")
        f.write(f"Race/Ethnicity: {user.persona['race/ethnicity']}\n")
        f.write(f"Education: {user.persona['education']}\n")
        f.write(f"Occupation: {user.persona['occupation']}\n")
        f.write(f"Work Schedule: {user.persona['work_schedule']}\n")
        f.write(f"Marital Status: {user.persona['marital_status']}\n")
        
        # User interests
        f.write("\nUser Interests:\n")
        f.write(f"{user.persona['interests']}\n")
        
        # User model
        f.write(f"\nLLM Model: {user.model}\n")
        
        # Treatment group
        f.write(f"\nTreatment Group: {'Yes' if user.treatment == 1 else 'No'}\n")
        if user.treatment == 1 and hasattr(user, 'treatment_type') and user.treatment_type:
            f.write(f"Treatment Type: {user.treatment_type}\n")
        
        # Voting history and outcome
        f.write("\nVoting History:\n")
        for round_num, likelihood in sorted(user.voting_history.items()):
            f.write(f"Round {round_num}: {likelihood} out of 4\n")
            
        # Add voting outcome
        if hasattr(user, 'voted') and user.voted is not None:
            f.write(f"\nVoting Outcome: {'Voted' if user.voted == 1 else 'Did not vote'}\n")
        else:
            f.write("\nVoting Outcome: Undetermined\n")
        
        # Social network - following
        f.write("\nFollowing Users:\n")
        for followed_id in sorted(user.following):
            if followed_id in agents:
                followed = agents[followed_id]
                f.write(f"User {followed_id}: {followed.persona['age']} year old {followed.persona['gender']}, {followed.persona['occupation']}, Model: {followed.model}\n")
            else:
                f.write(f"User {followed_id}: User not found in agents\n")
        
        # Social network - followers
        f.write("\nFollowers:\n")
        for follower_id in sorted(user.followers):
            if follower_id in agents:
                follower = agents[follower_id]
                f.write(f"User {follower_id}: {follower.persona['age']} year old {follower.persona['gender']}, {follower.persona['occupation']}, Model: {follower.model}\n")
            else:
                f.write(f"User {follower_id}: User not found in agents\n")
        
        # Add all message exchanges between simulator and agent
        f.write("\n=== Conversation History ===\n\n")
        for i, message in enumerate(user.full_conversation_history):
            role = "Simulator" if message['role'] == 'user' else "Agent"
            
            # Check if this is a voting decision message
            is_voting_msg = "Election day has passed" in message.get('content', '') or \
                           (role == "Agent" and len(message.get('content', '').strip()) <= 5 and \
                           any(x in message.get('content', '') for x in ['0', '1']))
            
            # Add a special header for the voting decision message
            if is_voting_msg and role == "Simulator":
                f.write(f"Message {i+1} ({role} - VOTING DECISION QUERY):\n")
            elif is_voting_msg and role == "Agent":
                voted = "VOTED" if "1" in message.get('content', '')[:10] else "DID NOT VOTE"
                f.write(f"Message {i+1} ({role} - {voted}):\n")
            else:
                f.write(f"Message {i+1} ({role}):\n")
                
            f.write(f"{message['content']}\n")
            f.write("\n" + "-"*50 + "\n\n")
        
        # Add content interactions
        f.write("=== Content Interactions ===\n\n")
        for content_id in sorted(user.seen_content):
            try:
                content = next(c for c in contents if c['content_id'] == content_id)
                
                f.write(f"Content ID: {content_id}\n")
                f.write(f"Author: User {content['author_id']}\n")
                f.write(f"Topic: {content['topic']} - {content['subtopic']}\n")
                f.write(f"Content: {content['text']}\n")
                f.write(f"Rounds Seen: {', '.join(map(str, sorted(user.content_seen_rounds[content_id])))}\n")
                
                # Check if user liked this content
                if content_id in user.likes:
                    f.write("✓ Liked this content\n")
                
                # Check if user replied to this content
                if content_id in user.replies:
                    f.write(f"Reply: {user.replies[content_id]}\n")
                
                # Show all replies to this content
                if content['replies']:
                    f.write("\nReplies to this content:\n")
                    for reply in content['replies']:
                        f.write(f"User {reply['user_id']}: {reply['text']}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
            except StopIteration:
                f.write(f"Content ID: {content_id} - This content no longer exists in the system\n\n")
                continue

def determine_voting_outcome(agent, election_day):
    """
    Determines if an agent actually voted in the election.
    
    Args:
        agent: The agent to check
        election_day: The day of the election
    
    Returns:
        1 if the agent voted, 0 if they did not
    """
    # Users under 18 cannot vote
    if agent.persona['age'] < 18:
        return 0
        
    # Use the most recent voting likelihood as a baseline
    latest_likelihood = 0
    for round_num in sorted(agent.voting_history.keys()):
        if round_num <= election_day:
            latest_likelihood = agent.voting_history[round_num]
    
    # Prepare the prompt for the LLM
    persona_description = assign_persona_to_model(
        agent.persona, [
            'gender',
            'age',
            'race/ethnicity',
            'education',
            'occupation',
            'marital_status',
            'work_schedule',
            'native_country',
            'interests',
            'close_friends',
        ])
    political_stance = agent.persona['political_stance']

    prompt = f"{persona_description}\n\n"
    prompt += f"This is the election day."
    
    prompt += f" Your most recent voting likelihood was {latest_likelihood}/4, where:\n"
    prompt += "0 - Definitely will not vote\n"
    prompt += "1 - Probably will not vote\n"
    prompt += "2 - Might or might not vote\n"
    prompt += "3 - Probably will vote\n"
    prompt += "4 - Definitely will vote\n\n"

    prompt += f"Considering your **voting likelihood of {latest_likelihood}/4** and your political stance ({political_stance}),"
    prompt += " reply with ONLY '1' if you voted or '0' if you did not vote."
    
    # Call the LLM to decide
    messages = [{'role': 'user', 'content': prompt}]
    response = gen_completion(messages, model=agent.model)
    
    # Add the voting decision conversation to the agent's history
    messages.append({'role': 'assistant', 'content': response})
    agent.full_conversation_history.extend(messages)
    
    # Extract just the first character and convert to int if possible
    try:
        if '1' in response[:10]:
            return 1
        else:
            return 0
    except:
        # Default to using likelihood as probability if response parsing fails
        probability = latest_likelihood / 4.0
        return 1 if random.random() < probability else 0

def save_network_data(agents, output_file):
    """
    Save the follower-following network to a CSV file.
    
    Args:
        agents: Dictionary of all agents
        output_file: Path to save the network data
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target'])  # Header row
        
        # Write all follower-following relationships
        for source_id, agent in agents.items():
            for target_id in agent.following:
                writer.writerow([source_id, target_id])
    
    # Also save as a NetworkX graph for easier analysis
    G = nx.DiGraph()
    for source_id, agent in agents.items():
        G.add_node(source_id, user_id=source_id)  # Add user ID as a node attribute
        for target_id in agent.following:
            G.add_edge(source_id, target_id)
    
    # Convert output_file to string for the next line
    output_file_str = str(output_file)
    
    # Save graph in GraphML format
    nx.write_graphml(G, f"{output_file_str.rsplit('.', 1)[0]}.graphml")

def calculate_and_print_turnout(agents, setting_name=None):
    """
    Calculate and print election turnout statistics.
    
    Args:
        agents: Dictionary of all agents
        setting_name: Name of the treatment setting (optional)
    
    Returns:
        dict: Dictionary containing turnout statistics
    """
    # Count eligible voters (age >= 18)
    eligible_voters = [agent for agent in agents.values() if agent.persona['age'] >= 18]
    total_eligible = len(eligible_voters)
    
    if total_eligible == 0:
        print("No eligible voters found (all users under 18)")
        return {"total_eligible": 0, "total_voted": 0, "turnout_rate": 0}
    
    # Count who actually voted
    voted_count = sum(1 for agent in eligible_voters if hasattr(agent, 'voted') and agent.voted == 1)
    
    # Calculate turnout rate
    turnout_rate = (voted_count / total_eligible) * 100
    
    # Print results
    if setting_name:
        print(f"\n=== ELECTION TURNOUT for {setting_name} ===")
    else:
        print(f"\n=== ELECTION TURNOUT ===")
    
    print(f"Total eligible voters (age >= 18): {total_eligible}")
    print(f"Total voters who voted: {voted_count}")
    print(f"Turnout rate: {turnout_rate:.1f}%")
    
    # Breakdown by treatment if applicable
    treatment_stats = {"control": {"eligible": 0, "voted": 0}, "treatment": {"eligible": 0, "voted": 0}}
    
    for agent in eligible_voters:
        group = "treatment" if hasattr(agent, 'treatment') and agent.treatment == 1 else "control"
        treatment_stats[group]["eligible"] += 1
        if hasattr(agent, 'voted') and agent.voted == 1:
            treatment_stats[group]["voted"] += 1
    
    # Print treatment breakdown if there are both control and treatment groups
    if treatment_stats["control"]["eligible"] > 0 and treatment_stats["treatment"]["eligible"] > 0:
        control_turnout = (treatment_stats["control"]["voted"] / treatment_stats["control"]["eligible"]) * 100 if treatment_stats["control"]["eligible"] > 0 else 0
        treatment_turnout = (treatment_stats["treatment"]["voted"] / treatment_stats["treatment"]["eligible"]) * 100 if treatment_stats["treatment"]["eligible"] > 0 else 0
        
        print(f"\nBreakdown by group:")
        print(f"Control group: {treatment_stats['control']['voted']}/{treatment_stats['control']['eligible']} ({control_turnout:.1f}%)")
        print(f"Treatment group: {treatment_stats['treatment']['voted']}/{treatment_stats['treatment']['eligible']} ({treatment_turnout:.1f}%)")
    
    print("=" * 40)
    
    return {
        "total_eligible": total_eligible,
        "total_voted": voted_count,
        "turnout_rate": turnout_rate,
        "control_stats": treatment_stats["control"],
        "treatment_stats": treatment_stats["treatment"]
    }