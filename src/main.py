"""
Main entry point for the LLM election simulation.
This file reexports the main simulation functionality from the appropriate modules.
"""
import os
import random
import pandas as pd
import numpy as np
import argparse
import json
import networkx as nx
from collections import defaultdict
import copy
from multiprocessing import Pool, cpu_count
from functools import partial
import re

# Import configuration parameters
from src.config import (
    DEFAULT_MODEL, rate_41nano, rate_41mini, rate_41,
    context_history_length, content_gen_prob, max_activity_interval,
    p_interest, p_followed, p_trending, p_random,
    PATH_TO_DEMOGRAPHC_DATA, PATH_TO_USER_DATA
)

# Import from modules
from src.simulation.simulation import run_simulation, process_agent, select_feed_content, generate_content
from src.simulation.run_multiple import run_multiple_simulations
from src.simulation.simulation_utils import save_population_summary, save_user_interactions, determine_voting_outcome, save_network_data
from src.agents.persona import load_profiles_and_network, save_all_personas, assign_persona_to_model, calculate_model_probabilities
from src.agents.agent import Agent
from src.models.feed_ranking import FeedRankingAlgorithm
from src.utils.llm_utils import gen_completion

# Export all functions and classes for backward compatibility
__all__ = [
    # Simulation functions
    'run_simulation',
    'run_multiple_simulations',
    'process_agent',
    'select_feed_content',
    'generate_content',
    
    # Utility functions
    'save_population_summary',
    'save_user_interactions',
    'save_all_personas',
    'save_network_data',
    
    # Agent-related functions
    'load_profiles_and_network',
    'assign_persona_to_model',
    'calculate_model_probabilities',
    'determine_voting_outcome',
    
    # Classes
    'Agent',
    'FeedRankingAlgorithm',
    
    # LLM utilities
    'gen_completion',
    
    # Configuration parameters
    'DEFAULT_MODEL',
    'rate_41nano',
    'rate_41mini',
    'rate_41',
    'context_history_length',
    'content_gen_prob',
    'max_activity_interval',
    'p_interest',
    'p_followed',
    'p_trending',
    'p_random',
    'PATH_TO_DEMOGRAPHC_DATA',
    'PATH_TO_USER_DATA'
] 