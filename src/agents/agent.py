"""
Agent class and related functionality for the election simulation.
"""
import random
import json
from collections import defaultdict
import re

from src.utils.llm_utils import gen_completion
from src.config import (
    context_history_length, max_activity_interval
)

def assign_persona_to_model(persona, demos_to_include):
    """
    Describe persona in second person: "You are..."
    """
    # Start with basic demographics
    s = f"You are, as a social-network user, a {persona['age']}y {persona['race/ethnicity']} {persona['gender']} from {persona['native_country']}, "

    # Add work schedule and marital status
    s += f"{persona['work_schedule']}, {persona['marital_status'].lower()}, "
    
    # Add education and occupation context
    s += f"{persona['education']}, {persona['occupation']}, "
    
    # Add interests in a natural way
    s += f"interests:{persona['interests']}. "
    
    # Add close friends in a natural way
    s += f"You are close-friend with Users:{persona['close_friends']}. "
    
    return s

class Agent:
    def __init__(self, user_id, persona):
        self.user_id = user_id
        self.persona = persona
        self.model = persona['model']  # Use the model specified in the persona
        self.following = []  # Users this agent follows
        self.followers = []  # Users following this agent
        self.engagement_history = defaultdict(list)  # content_id: list of rounds engaged
        self.content_seen_rounds = defaultdict(list)  # content_id: list of all rounds when content was seen
        self.likes = set()  # content_ids liked
        self.replies = {}  # content_id: reply text
        self.treatment = None  # Will be set later
        self.treatment_type = None  # Type of treatment (info message or social message)
        self.likely_voter_data = None  # Data about likely voters {count: int, users: []}
        self.next_activity_time = None  # Will be set when agent becomes active

        self.seen_content = set()  # content_ids the agent has already seen
        self.content_last_seen_replies = {}  # content_id: number of replies last time agent saw it

        self.conversation_history = []  # list of messages for LLM context (limited)
        self.full_conversation_history = []  # list of all messages for tracking (unlimited)

        self.friend_like_counts = defaultdict(int)
        self.friend_reply_counts = defaultdict(int)
        
        # Voting attributes
        self.voting_likelihood = None  # Scale 0-4, where 0 is not voting, 4 is definitely voting
        self.voting_history = {}  # round_num: voting_likelihood
        self.voted = None  # 1 if agent voted, 0 if not, None if undetermined

    def decide_engagement(self, feed_contents, round_num, election_day):
        # Include persona details
        persona_description = assign_persona_to_model(
            self.persona, [
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
        political_stance = self.persona['political_stance']

        # Start constructing the prompt
        prompt = f"{persona_description}\nRound:{round_num}\n"

        # Add previous activity context if available
        if self.conversation_history:
            last_round = None
            last_activity = None
            last_feed_contents = None
            for msg, last_fc, last_rn in reversed(self.conversation_history):
                last_round = last_rn
                last_feed_contents = last_fc
                try:
                    result = json.loads(msg['content'])
                    if 'engagement' in result:
                        last_activity = result
                        break
                except:
                    continue
                
            if last_round is not None and last_activity is not None:
                # Check if there was any engagement
                has_engagement = any(decision.get('engage', 'nothing') != 'nothing' for decision in last_activity['engagement'])
                
                if has_engagement:
                    prompt += f"Last Round:{last_round}\nLast Engagement:\n"
                    for decision in last_activity['engagement']:
                        post_num = decision.get('post_number', 0)
                        engage = decision.get('engage', 'nothing')
                        if engage != 'nothing':  # Only include posts that were engaged with
                            if 0 <= post_num - 1 < len(last_feed_contents):
                                content = last_feed_contents[post_num - 1]
                                prompt += f"Post:{content['text']}\nAction:{engage}"
                                if engage == 'reply':
                                    prompt += f"\nReply:{decision.get('reply_text', '')}\n"
                                else:
                                    prompt += "\n"
                else:
                    prompt += f"Last Round:{last_round}\nLast Action:None\n"

                # Include previous voting decision if available
                if 'voting_likelihood' in last_activity:
                    prompt += f"Your previous voting likelihood: {last_activity['voting_likelihood']}/4\n"

        prompt += "\nFeed:\n"

        # For each content in feed_contents, add to prompt
        feed_topics = []
        
        for idx, content in enumerate(feed_contents):
            position_in_feed = idx + 1  # Positions start from 1
            content_text = content['text']
            content_id = content['content_id']
            author_id = content.get('author_id') or content.get('user_id')
            is_followed = "Yes" if author_id in self.following else "No"
            
            # Get the author's public profile from the content
            author_profile = "Unknown"
            if 'author_profile' in content:
                author_profile = content['author_profile']
            
            # Collect topics from feed for potential content generation
            topic = content.get('topic', 'general')
            subtopic = content.get('subtopic', 'general')
            feed_topics.append((topic, subtopic))

            # Include only the most recent reply
            replies_text = ""
            if content.get('replies'):  # Check if there are any replies
                if isinstance(content['replies'], list) and content['replies']:
                    recent_reply = content['replies'][-1]  # Get the last reply
                    user_id = recent_reply.get('user_id', 'unknown')
                    text = recent_reply.get('text', '')
                    replies_text = f"Reply: User{user_id}:{text[:140]}"
                elif isinstance(content['replies'], dict) and content['replies']:
                    # Get the last key-value pair from the dictionary
                    user_id, text = list(content['replies'].items())[-1]
                    replies_text = f"Reply: User{user_id}:{text[:140]}"

            friend_like_count = self.friend_like_counts.get(content_id, 0)
            friend_reply_count = self.friend_reply_counts.get(content_id, 0)

            prompt += f"Post:{position_in_feed}\nAuthor:{author_profile}\nFollowed:{is_followed}\nTopic:{topic}-{subtopic}\nFollowing Likes:{friend_like_count}\nFollowing Replies:{friend_reply_count}\nTotal Replies:{len(content.get('replies', []))}\nText:{content_text}\n{replies_text if replies_text else ''}\n"

        # Add option to generate content with probability content_gen_prob
        from src.config import content_gen_prob
        generate_content = random.random() < content_gen_prob
        main_topic = ""
        subtopic = ""
        
        if generate_content:
            # Use agent's interests to guide content creation
            user_interests = self.persona['interests'].split(',') if ',' in self.persona['interests'] else [self.persona['interests']]
            
            # Either use a topic from the feed or one of the agent's interests
            if feed_topics and random.random() < 0.7:  # 70% chance to be inspired by feed
                main_topic, subtopic = random.choice(feed_topics)
            else:  # 30% chance to create content based on own interests
                # Use a generic topic or a topic from another post if available
                main_topic = feed_topics[0][0] if feed_topics else "general"
                subtopic = random.choice(user_interests)
            
            # Make election/politics topics more realistic with specificity
            if main_topic.lower() == "politics":
                # Create more specific political topics that would appear during an election
                political_subtopics = [
                    "candidate positions", "campaign updates", "voting deadlines", 
                    "election polls", "debate highlights", "policy issues",
                    "voter registration", "fact checking", "campaign rallies",
                    "election news", "voting rights", "candidate endorsements"
                ]
                subtopic = random.choice(political_subtopics) if random.random() < 0.7 else subtopic
                
                # Add realistic election context to content generation
                days_str = "today" if election_day - round_num == 0 else (
                    "tomorrow" if election_day - round_num == 1 else f"in {election_day - round_num} days"
                )
                
                if election_day - round_num <= 7:  # Final week
                    prompt += f"\nThe election is {days_str}! As a {self.persona['political_stance']} voter, you might consider posting about {subtopic}."
                else:
                    prompt += f"\nWith the election {days_str}, as a {self.persona['political_stance']} voter, you might consider sharing your thoughts on {subtopic}."
            else:
                prompt += f"\nYou may choose to create a new post about topic-{main_topic}, subtopic-{subtopic}, or decline to post.\n"

        # Add election context only for users 18 and older
        if self.persona['age'] >= 18:
            # Add election voting context
            prompt += "\n"
            
            # Calculate days until election (election day is the last day of the main stage)
            days_until_election = max(0, election_day - round_num)
            
            if days_until_election == 0:
                prompt += "Today is election day! "
            elif days_until_election == 1:
                prompt += "Tomorrow is election day! "
            else:
                prompt += f"The election is in {days_until_election} days. "
            
            # Add encouraging message for treatment group
            if self.treatment == 1:
                # Check if this is a social message treatment type
                if hasattr(self, 'treatment_type') and (self.treatment_type == "treatment_soc_message" or 
                                                       self.treatment_type == "experiment_soc_message" or 
                                                       self.treatment_type == "control_then_soc"):
                    prompt += "VOTE OR BE SILENCED! One ballot = one voice. Use yours."
                    
                    # Only consider users that this agent follows
                    likely_voters = []
                    likely_voter_count = 0
                    
                    # Check which followed users are likely to vote (chose 3 or 4)
                    if hasattr(self, 'likely_voter_data') and isinstance(self.likely_voter_data, dict):
                        likely_voter_count = self.likely_voter_data.get('count', 0)
                        # Filter the likely voters to only include those the agent follows
                        likely_voters = [user_id for user_id in self.following if user_id in self.likely_voter_data.get('users', [])]
                        
                        # Display the number of likely voters
                        if likely_voter_count:
                            prompt += f" {likely_voter_count} users have indicated they're likely to vote."

                        # Display up to 6 likely voters that this agent follows
                        if likely_voters:
                            # Randomly select up to 6 likely voters to display
                            sample_size = min(6, len(likely_voters))
                            selected_voters = random.sample(likely_voters, sample_size)
                            
                            prompt += f" From users you follow, users {', '.join(map(str, selected_voters))} are planning to vote. Will you?"
                    
                else:
                    # Default treatment message (info only)
                    prompt += " VOTE OR BE SILENCED! One ballot = one voice. Use yours."
                        
            # Add a touch of realism instruction
            prompt += "\nAs someone with your background and values, consider realistically how likely you are to vote in the upcoming election:\n"
            prompt += "On a scale from 0 to 4, where:\n"
            prompt += "0 - Definitely will not vote\n"
            prompt += "1 - Probably will not vote\n"
            prompt += "2 - Might or might not vote\n"
            prompt += "3 - Probably will vote\n"
            prompt += "4 - Definitely will vote\n"
            
            # Include previous voting decision if available
            if self.voting_history and round_num > 1:
                prev_rounds = sorted([r for r in self.voting_history.keys() if r < round_num], reverse=True)
                if prev_rounds:
                    last_round = prev_rounds[0]
                    last_likelihood = self.voting_history[last_round]
                    prompt += f"\nIn your last response (round {last_round}), your voting likelihood was: {last_likelihood}/4."
        
        # Add instructions with clear JSON formatting requirements
        prompt += """\n\nGiven above posts, decide which posts to engage with. Reply ONLY in valid JSON format, NO additional text or comments.

Output must precisely match this schema:
{
  "engagement": [
    {
      "post_number": <int>,
      "engage": "nothing"|"like"|"reply",
      "reply_text": <string>,
      "follow_action": "follow"|"unfollow"|"no_change"
    }, …
  ],"""
        
        if generate_content:
            prompt += """
  "generated_content": <string>, // max 140chars"""
        
        if self.persona['age'] >= 18:
            prompt += """
  "voting_likelihood": <int>, // Must be 0-4 integer **based on your POLITICAL STANCE: """ + political_stance + """**"""
        
        prompt += """
  "next_activity_time": <int> // Must be between 0 and """ + str(max_activity_interval+1) + """ based on your persona and engagement
        
}

CRITICAL RULES:
1.Use DOUBLE QUOTES for ALL keys and string values
2.Do NOT use any escape characters like \\, \\" or \\'
3.Include "reply_text" field ONLY when "engage" is "reply"
4.Include "follow_action" for each post to decide whether to follow/unfollow the author, if you like to see more/less from them
5.Ensure ALL JSON is properly terminated with closing brackets      
6.STRONGLY prefer "like" over "reply", Very RARELY use "reply"
7.Posts higher in the feed, higher post-sentiment, with engagement from users you follow, or from authors you follow → MORE LIKELY TO ENGAGE
8.Your response MUST be ONLY the JSON object, nothing else"""

        if generate_content:
            prompt += """
9."generated_content" must be a social media post related to your personality and seen posts, or can be empty string "" if you choose not to post
            """

        prompt += "\n**React as a real social media user would with your persona, political stance, social ties, and values.**"

        # Initialize response variables
        max_retries = 2
        response = None
        parsed_json = None
        
        # Try up to max_retries times to get a valid JSON response
        for attempt in range(max_retries + 1):
            # Prepare conversation history
            messages = []
            messages.append({'role': 'user', 'content': prompt})
            
            try:
                # Use LLM to decide engagement
                response = gen_completion(messages, model=self.model)
                
                # Extract the JSON portion if there's extra text
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    
                    # Simple cleanup for common issues
                    json_str = json_str.replace('\\"', '"')
                    json_str = json_str.replace("\\'", "'")
                    
                    # Try to parse the JSON
                    parsed_json = json.loads(json_str)
                    
                    # If we got here, parsing succeeded
                    break
                    
            except Exception as e:
                if attempt < max_retries:
                    # Add clarification for retry
                    retry_prompt = f"Your previous response could not be parsed as valid JSON. Please provide ONLY a properly formatted JSON object according to the schema. Error: {str(e)[:100]}"
                    messages.append({'role': 'assistant', 'content': response})
                    messages.append({'role': 'user', 'content': retry_prompt})
                    prompt = retry_prompt
                else:
                    print(f"All retry attempts failed for agent {self.user_id}: {e}")
        
        # If we still don't have valid JSON, create a minimal default response
        if parsed_json is None:
            parsed_json = {
                "engagement": [],
                "voting_likelihood": random.randint(0, 4) if self.persona['age'] >= 18 else 0,
                "next_activity_time": random.randint(1, max_activity_interval)
            }
            if generate_content:
                parsed_json["generated_content"] = f"Post about {main_topic}-{subtopic}"
            
            # Use the default JSON for the response
            response = json.dumps(parsed_json)
        
        # Update both conversation histories
        messages.append({'role': 'assistant', 'content': response})
        self.conversation_history = [(msg, feed_contents, round_num) for msg in messages[-context_history_length:]]
        self.full_conversation_history.extend(messages)

        # Process the parsed response
        engagement_decisions = parsed_json.get("engagement", [])
        generated_content_text = parsed_json.get("generated_content", "")
        voting_likelihood = parsed_json.get("voting_likelihood", random.randint(0, 4))
        next_activity_time = parsed_json.get("next_activity_time", random.randint(1, max_activity_interval))
        
        # Ensure voting_likelihood is valid for users 18 and older, set to 0 for users under 18
        if self.persona['age'] < 18:
            voting_likelihood = 0
        elif not isinstance(voting_likelihood, int) or voting_likelihood < 0 or voting_likelihood > 4:
            voting_likelihood = random.randint(0, 4)
            
        # Update voting likelihood and history only for users 18 and older
        if self.persona['age'] >= 18:
            self.voting_likelihood = voting_likelihood
            self.voting_history[round_num] = voting_likelihood
        
        # Ensure next_activity_time is valid
        if not isinstance(next_activity_time, int) or next_activity_time < 1 or next_activity_time > max_activity_interval:
            next_activity_time = random.randint(1, max_activity_interval)
        
        # Set next activity time
        # Introduce randomness to next_activity_time
        random_offset = random.randint(0, 1 + int(next_activity_time))  # Random offset
        self.next_activity_time = round_num + next_activity_time + random_offset
        
        # Return a structured dictionary with engagement decisions and voting likelihood 
        return {
            "engagement": engagement_decisions,
            "generated_content": generated_content_text,
            "voting_likelihood": voting_likelihood
        }

    def update_friend_engagement_counts(self, agents):
        """
        Updates the counts of likes and replies by the agent's friends.
        
        Args:
            agents: Dictionary of all agents
        """
        self.friend_like_counts = defaultdict(int)
        self.friend_reply_counts = defaultdict(int)
        
        for friend_id in self.following:
            if friend_id in agents:
                friend = agents[friend_id]
                for content_id in friend.likes:
                    self.friend_like_counts[content_id] += 1
                for content_id in friend.replies.keys():
                    self.friend_reply_counts[content_id] += 1 