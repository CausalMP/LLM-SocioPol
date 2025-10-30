"""
Module for running multiple simulations with different settings.
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import copy
import random
import time
import networkx as nx
import csv
from datetime import datetime, timedelta
from multiprocessing import cpu_count

# Import simulation functions
from src.simulation.simulation import run_simulation
from src.simulation.simulation_utils import save_user_interactions, save_network_data
from src.agents.persona import save_all_personas
from src.config import time_points, warmup_periods, treatment_settings

def format_time(seconds):
    """Format seconds into a human-readable string."""
    return str(timedelta(seconds=int(seconds)))

def run_multiple_simulations(N_runs, n_users, num_contents, feed_length, topic, model, n_cores=None, random_seed=None, treatment_seed=None):
    """
    Run multiple simulations with different treatment probabilities.
    
    Args:
        N_runs (int): Number of runs for each treatment setting
        n_users (int): Number of users in the simulation
        num_contents (int): Number of content pieces
        feed_length (int): Length of each user's feed
        topic (str): Main topic for content
        model (str): LLM model to use
        n_cores (int, optional): Number of CPU cores to use for parallel processing. If None, uses all available cores.
        random_seed (int, optional): Random seed for general simulation randomness
        treatment_seed (int, optional): Random seed for treatment allocation only. If None, uses random_seed
    """
    # Start timing
    start_time = time.time()
    print(f"\nStarting multiple simulations at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters: N_runs={N_runs}, n_users={n_users}, num_contents={num_contents}, feed_length={feed_length}")
    if n_cores is None:
        n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    # Set election day as the max time point
    election_day = max(time_points)
    print(f"Election day set to day {election_day}")
    
    # Create output directory if it doesn't exist
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run N_runs simulations
    for run in range(N_runs):
        run_start_time = time.time()
        print(f"\nRunning simulation {run + 1}/{N_runs}")
        
        # Set random seed for this run
        run_seed = random_seed + run if random_seed is not None else None
        
        # Set treatment seed for this run
        run_treatment_seed = treatment_seed + run if treatment_seed is not None else run_seed
        
        # Create directory for this run
        run_dir = Path(output_dir) / f"seed{run}"
        run_dir.mkdir(exist_ok=True)
        
        # Run warmup period with negative round numbers
        print("Running warmup period...")
        warmup_engagement_data, warmup_treatment_data, warmup_activity_data, warmup_voting_data, warmup_state = run_simulation(
            n_users=n_users,
            time_points=[0],  # Last warmup round will be 0
            treatment_probs=[0],  # No treatment during warmup
            topic=topic,
            c=num_contents,
            feed_length=feed_length,
            random_seed=run_seed,
            treatment_seed=run_treatment_seed,  # Use treatment seed for this run
            is_warmup=True,
            output_dir=str(run_dir),
            start_round=-warmup_periods,  # Start from negative number
            n_cores=n_cores,  # Pass the number of cores
            election_day=election_day 
        )
        
        # Run each treatment setting for this run, starting from the warmup state
        for treatment_probs, setting_name in treatment_settings:
            setting_start_time = time.time()
            print(f"Treatment setting: {setting_name}")
            
            # Run simulation with warmup state
            engagement_data, treatment_data, activity_data, voting_data, current_state = run_simulation(
                n_users=n_users,
                time_points=time_points,
                treatment_probs=treatment_probs,
                topic=topic,
                c=num_contents,
                feed_length=feed_length,
                random_seed=run_seed,
                treatment_seed=run_treatment_seed,  # Use treatment seed for this run
                initial_state=warmup_state, 
                n_cores=n_cores,  # Pass the number of cores
                election_day=election_day,  # Pass election day directly
                setting_name=setting_name  # Pass the setting name
            )
            
            # Combine warmup data with the current setting's data
            # For panel data
            combined_engagement_data = pd.concat([warmup_engagement_data, engagement_data], axis=1)
            # For treatment data
            combined_treatment_data = pd.concat([warmup_treatment_data, treatment_data], axis=1)
            # For activity data
            combined_activity_data = pd.concat([warmup_activity_data, activity_data], axis=1)
            # For voting data
            combined_voting_data = pd.concat([warmup_voting_data, voting_data], axis=1)
            
            # File paths for saving results
            engagement_data_file = run_dir / f"{setting_name}_n{n_users}_t{election_day}_seed{run}_engagement_data.csv"
            treatment_data_file = run_dir / f"{setting_name}_n{n_users}_t{election_day}_seed{run}_treatment_data.csv"
            activity_data_file = run_dir / f"{setting_name}_n{n_users}_t{election_day}_seed{run}_activity_data.csv"
            voting_data_file = run_dir / f"{setting_name}_n{n_users}_t{election_day}_seed{run}_voting_data.csv"
            personas_file = run_dir / f"{setting_name}_seed{run}_personas.csv"
            
            # Save combined results
            combined_engagement_data.to_csv(engagement_data_file)
            combined_treatment_data.to_csv(treatment_data_file)
            combined_activity_data.to_csv(activity_data_file)
            combined_voting_data.to_csv(voting_data_file)
            
            # Save users' interactions
            run_dir_ui = Path(output_dir) / f"seed{run}" / f"{setting_name}"
            run_dir_ui.mkdir(exist_ok=True)
            for user_id in current_state['agents']:
                save_user_interactions(
                    user_id,
                    current_state['agents'],
                    current_state['contents'],
                    run_dir_ui / f"user_{user_id}_interactions.txt"
                )
            
            # Save all personas for this treatment setting
            save_all_personas(
                current_state['agents'],
                personas_file
            )
            
            setting_time = time.time() - setting_start_time
            print(f"Completed {setting_name} setting in {format_time(setting_time)}")
        
        run_time = time.time() - run_start_time
        print(f"Completed run {run + 1} in {format_time(run_time)}")
        
        # Save network data
        save_network_data(
            current_state['agents'],
            run_dir / f"seed{run}_network.csv"
        )
    
    # Calculate and print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {format_time(total_time)}")
    print(f"Average time per run: {format_time(total_time/N_runs)}")
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    parser = argparse.ArgumentParser(description='Run multiple social media simulations with different treatment settings.')
    parser.add_argument('--N_runs', type=int, default=1, help='Number of runs for each treatment setting')
    parser.add_argument('--n_users', type=int, default=3, help='Number of users')
    parser.add_argument('--num_contents', type=int, default=20, help='Number of content pieces')
    parser.add_argument('--feed_length', type=int, default=4, help='Feed length')
    parser.add_argument('--topic', type=str, default='Politics', help='Topic string')
    parser.add_argument('--model', type=str, default='gpt-4.1-nano', help='LLM model to use')
    parser.add_argument('--n_cores', type=int, default=4, help='Number of CPU cores to use for parallel processing. If not specified, uses all available cores.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for general simulation randomness')
    parser.add_argument('--treatment_seed', type=int, default=None, help='Random seed for treatment allocation only. If not specified, uses random_seed')
    args = parser.parse_args()
    
    run_multiple_simulations(
        N_runs=args.N_runs,
        n_users=args.n_users,
        num_contents=args.num_contents,
        feed_length=args.feed_length,
        topic=args.topic,
        model=args.model,
        n_cores=args.n_cores,
        random_seed=args.random_seed,
        treatment_seed=args.treatment_seed
    )

if __name__ == "__main__":
    main() 