"""
Persona generation and management functions for the election simulation.
"""
import os
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Import from config module
from src.config import (
    DEFAULT_MODEL, rate_41nano, rate_41mini, rate_41,
    PATH_TO_DEMOGRAPHC_DATA, PATH_TO_USER_DATA
)

def load_profiles_and_network(n_users=None, random_seed=None):
    """
    Load agent profiles and network connections from CSV files.
    
    Args:
        n_users: Optional number of users to limit to (randomly sampled)
        random_seed: Random seed for reproducibility
    
    Returns:
        A tuple containing (personas, network_edges)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # Load profiles from CSV
    profiles_df = pd.read_csv(os.path.join(PATH_TO_USER_DATA, 'profiles.csv'))
    
    # Sample profiles if n_users is specified
    if n_users is not None and n_users < len(profiles_df):
        profiles_df = profiles_df.sample(n=n_users, random_state=random_seed)
    
    # Load network from CSV
    network_df = pd.read_csv(os.path.join(PATH_TO_USER_DATA, 'network.csv'))
    
    # Ensure we only include network connections for the sampled users
    if n_users is not None:
        user_ids = set(profiles_df['user_id'].values)
        network_df = network_df[
            (network_df['source_user_id'].isin(user_ids)) & 
            (network_df['target_user_id'].isin(user_ids))
        ]
    
    # Convert profiles to personas dictionary
    personas = {}
    for _, row in profiles_df.iterrows():
        user_id = row['user_id']
        
        # Extract interests as a list
        interests_str = row.get('interests', '')
        if pd.isna(interests_str) or interests_str == '':
            interests = ['Politics', 'Technology']  # Default interests
        else:
            interests = [i.strip() for i in interests_str.split(',')]
            # Ensure we have at least 2 interests
            if len(interests) < 2:
                interests.append('Technology' if 'Technology' not in interests else 'Politics')
        
        # Use the model from profiles if available, otherwise assign based on education/age
        model = row.get('model', '')
        if pd.isna(model) or model == '':
            # Calculate model probabilities based on user characteristics
            model_probs = calculate_model_probabilities(
                education=row['education'],
                occupation=row['occupation'],
                age=row['age']
            )
            # Select model based on probabilities
            models = list(model_probs.keys())
            model = np.random.choice(
                models,
                p=[model_probs[m] for m in models]
            )
        
        # Convert work hours to schedule if needed
        work_schedule = row.get('work_schedule', '')
        if pd.isna(work_schedule) or work_schedule == '':
            # Convert hours_per_week to a descriptive work schedule
            hours = row['hours_per_week']  # Default to 40 if not available
            if hours <= 20:
                work_schedule = "part-time worker with flexible hours"
            elif hours <= 35:
                work_schedule = "part-time worker with regular hours"
            elif hours <= 45:
                work_schedule = "full-time worker with standard hours"
            elif hours <= 55:
                work_schedule = "full-time worker with long hours"
            else:
                work_schedule = "full-time worker with very long hours"
        
        # Create the persona dictionary
        persona = {
            'age': row['age'],
            'gender': row['sex'],
            'race/ethnicity': row['race'],
            'education': row['education'],
            'occupation': row['occupation'],
            'workclass': row['workclass'],
            'marital_status': row['marital_status'],
            'relationship': row['relationship'],
            'native_country': row['native_country'],
            'political_stance': row['political_stance'],
            'interests': row['interests'],
            'close_friends': row['close_friends'],
            'public_profile': row['public_profile'],
            'work_schedule': work_schedule,
            'model': model
        }
        personas[user_id] = persona
    
    return personas, network_df

def calculate_model_probabilities(education, occupation, age):
    """
    Calculate probabilities for different models based on user characteristics.
    Returns a dictionary of model:probability pairs.
    """
    # Base probabilities
    probs = {
        'gpt-4.1-nano': rate_41nano,
        'gpt-4.1-mini': rate_41mini,
        'gpt-4.1': rate_41
    }
    
    # Adjust probabilities based on education level
    if education in ['Doctorate']:
        probs['gpt-4.1'] *= 3.0
        probs['gpt-4.1-mini'] *= 1.5
        probs['gpt-4.1-nano'] *= 0.5
    elif education in ['Masters']:
        probs['gpt-4.1'] *= 2.0
        probs['gpt-4.1-mini'] *= 1.3
        probs['gpt-4.1-nano'] *= 0.7
    elif education in ['Bachelors']:
        probs['gpt-4.1'] *= 1.5
        probs['gpt-4.1-mini'] *= 1.2
        probs['gpt-4.1-nano'] *= 0.8
    
    # Adjust probabilities based on occupation
    if any(term in occupation.lower() for term in ['professor', 'researcher', 'scientist', 'engineer', 'analyst']):
        probs['gpt-4.1'] *= 2.0
        probs['gpt-4.1-mini'] *= 1.3
        probs['gpt-4.1-nano'] *= 0.7
    elif any(term in occupation.lower() for term in ['manager', 'director', 'specialist', 'consultant']):
        probs['gpt-4.1'] *= 1.5
        probs['gpt-4.1-mini'] *= 1.2
        probs['gpt-4.1-nano'] *= 0.8
    
    # Adjust probabilities based on age (older users tend to be more experienced)
    if age >= 40:
        probs['gpt-4.1'] *= 1.5
        probs['gpt-4.1-mini'] *= 1.2
        probs['gpt-4.1-nano'] *= 0.8
    elif age >= 30:
        probs['gpt-4.1'] *= 1.3
        probs['gpt-4.1-mini'] *= 1.1
        probs['gpt-4.1-nano'] *= 0.9
    
    # Normalize probabilities
    total = sum(probs.values())
    return {model: prob/total for model, prob in probs.items()}

def assign_persona_to_model(persona, demos_to_include):
    """
    Describe persona in second person: "You are..."
    """
    # Start with basic demographics
    s = f"You are a {persona['age']}y {persona['race/ethnicity']} {persona['gender']} from {persona['native_country']}, "

    # Add work schedule and marital status
    s += f"{persona['work_schedule']}, {persona['marital_status'].lower()}, "
    
    # Add education and occupation context
    s += f"{persona['education']}, {persona['occupation']}, "
    
    # Add interests in a natural way
    s += f"interests:{persona['interests']}. "
    
    # Add close friends in a natural way
    s += f"You are close-friend with Users:{persona['close_friends']}. "
    
    return s

def save_all_personas(agents, output_file):
    """
    Save detailed information about all agent personas to a CSV file.
    
    Args:
        agents: Dictionary of all agents
        output_file: Path to save the persona details
    """
    # Prepare a list to hold persona data
    persona_data = []

    for agent_id, agent in sorted(agents.items()):
        # Get voting outcome if available
        voted = agent.voted if hasattr(agent, 'voted') and agent.voted is not None else None
        
        # Create a dictionary for each agent's persona
        persona = {
            "user_id": agent_id,
            "age": agent.persona['age'],
            "gender": agent.persona['gender'],
            "race/ethnicity": agent.persona['race/ethnicity'],
            "education": agent.persona['education'],
            "occupation": agent.persona['occupation'],
            "work_schedule": agent.persona['work_schedule'],
            "marital_status": agent.persona['marital_status'],
            "political_stance": agent.persona['political_stance'],
            "interests": agent.persona['interests'],
            "close_friends": agent.persona['close_friends'],
            "llm_model": agent.model,
            "following_count": len(agent.following),
            "followers_count": len(agent.followers),
            "posts_liked_count": len(agent.likes),
            "posts_replied_to_count": len(agent.replies),
            "total_content_seen_count": len(agent.seen_content),
            "treatment_group": "Yes" if agent.treatment == 1 else "No", 
            "treatment_type": agent.treatment_type if hasattr(agent, 'treatment_type') and agent.treatment_type else "None",
            "voted": voted  # 1 if voted, 0 if not
        }
        persona_data.append(persona)

    # Create a DataFrame from the persona data
    df = pd.DataFrame(persona_data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False) 