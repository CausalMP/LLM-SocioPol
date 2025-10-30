"""
Main package initialization.
This module re-exports important functions and classes for backward compatibility.
"""

# Import Agent and related functions
from src.agents.agent import Agent, assign_persona_to_model
from src.agents.persona import load_profiles_and_network, save_all_personas

# Import simulation utilities
from src.simulation.simulation_utils import save_population_summary, save_user_interactions, determine_voting_outcome

# Import simulation functions
from src.simulation.simulation import run_simulation, process_agent

# Import feed ranking algorithm
from src.models.feed_ranking import FeedRankingAlgorithm

# Import LLM utilities
from src.utils.llm_utils import gen_completion

# Import configuration parameters
from src.config import (
    DEFAULT_MODEL, context_history_length, content_gen_prob, max_activity_interval,
    p_interest, p_followed, p_trending, p_random,
    PATH_TO_DEMOGRAPHC_DATA, PATH_TO_USER_DATA
)

# Global config parameters used in the original main.py
DEFAULT_MODEL = 'gpt-4.1-nano'
rate_41nano = 1.00
rate_41mini = 0.00
rate_41 = 0.00
context_history_length = 1
content_gen_prob = 0.05
max_activity_interval = 4

# Feed selection probabilities
p_interest = 0.60
p_followed = 0.20
p_trending = 0.15
p_random = 0.05

# Data paths
PATH_TO_DEMOGRAPHC_DATA = './us_demographics'
PATH_TO_USER_DATA = './users_data' 