"""
Configuration parameters for the election simulation.
"""

# Number of runs with different treatment seeds
n_runs = 1

# LLM model settings
DEFAULT_MODEL = 'gpt-4.1-nano'  # Use the appropriate model
rate_41nano = 0.98
rate_41mini = 0.016
rate_41 = 0.004
llm_temperature = 0.5       # Temperature for LLM calls (0.0-1.0, lower = more deterministic)
context_history_length = 1   # How many of the past feeds the API keeps in context
content_gen_prob = 0.01      # Chance of each user generating a new piece of content each round
max_activity_interval = 5    # Maximum number of rounds until next activity (update the guideline manually)

# Select content for the feed
p_interest = 0.60
p_followed = 0.20
p_trending = 0.15
p_random = 0.05

# Set the path for demographic data
PATH_TO_DEMOGRAPHC_DATA = './us_demographics'
PATH_TO_USER_DATA = './users_data'  # Path to the new CSV files 

# Simulation time settings
time_points = [10, 20, 29, 30]
warmup_periods = 15  # Number of warmup periods

# Random seed settings
random_seed = 44       # Default random seed for general simulation randomness
treatment_seed = 44    # Default random seed for treatment allocation only

# Treatment probability settings
treatment_settings = [
    # ([0.0, 0.0, 0.0, 0.0], "control"),
    # ([1.0, 1.0, 1.0, 1.0], "treatment_info_message"),
    ([0.2, 0.4, 0.8, 0.8], "experiment_info_message"),
    # ([1.0, 1.0, 1.0, 1.0], "treatment_soc_message"),
    # ([0.1, 0.2, 0.5, 0.5], "experiment_soc_message"),
    # ([0.0, 0.0, 0.0, 1.0], "control_then_info"),  # control until last time point, then info message
    # ([0.0, 0.0, 0.0, 1.0], "control_then_soc")    # control until last time point, then social message
] 