#!/usr/bin/env python
"""
Runner script for the warmup period only.
Runs the warmup stage and saves the state for use in later stages.
"""
import argparse
import pickle
import os
import psutil
import time
from pathlib import Path
from src.simulation.simulation import run_simulation, log_memory_usage
from src.simulation.simulation_utils import save_population_summary
from src.config import warmup_periods, random_seed, treatment_seed, time_points

def main():
    parser = argparse.ArgumentParser(description='Run the warmup period for a social media simulation with LLM agents.')
    parser.add_argument('--n_users', type=int, default=3, help='Number of users')
    parser.add_argument('--topic', type=str, default='Politics', help='Topic string')
    parser.add_argument('--num_contents', type=int, default=20, help='Number of content pieces')
    parser.add_argument('--feed_length', type=int, default=4, help='Feed length')
    parser.add_argument('--output_dir', type=str, default='simulation_results', help='Directory to save outputs')
    parser.add_argument('--n_cores', type=int, default=8, help='Number of CPU cores to use for parallel processing')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of users to process in one batch')
    
    args = parser.parse_args()
    
    # Print system information
    print(f"Total system memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the warmup directory path
    warmup_dir = Path(args.output_dir) / "warmup"
    warmup_dir.mkdir(exist_ok=True)
    
    # Calculate appropriate batch size based on user count
    recommended_batch_size = min(args.batch_size, max(100, args.n_users // 20))
    print(f"Using batch size of {recommended_batch_size} users")
    
    # Run warmup period with no treatments
    print("Running warmup period...")
    start_time = time.time()
    warmup_engagement_data, warmup_treatment_data, warmup_activity_data, warmup_voting_data, warmup_state = run_simulation(
        n_users=args.n_users,
        time_points=[0],  # Last warmup round will be 0
        treatment_probs=[0],  # No treatment during warmup
        topic=args.topic,
        c=args.num_contents,
        feed_length=args.feed_length,
        random_seed=random_seed,
        treatment_seed=treatment_seed,
        is_warmup=True,
        output_dir=str(warmup_dir),
        start_round=-warmup_periods,  # Start from negative number
        n_cores=args.n_cores,  # Pass the number of cores
        election_day=max(time_points), 
        batch_size=recommended_batch_size
    )
    warmup_time = time.time() - start_time
    print(f"Warmup simulation completed in {warmup_time:.2f} seconds")
    log_memory_usage("After warmup simulation")
    
    # Save warmup data
    print("Saving warmup data...")
    warmup_engagement_data.to_csv(warmup_dir / "engagement_data.csv")
    warmup_treatment_data.to_csv(warmup_dir / "treatment_data.csv")
    warmup_activity_data.to_csv(warmup_dir / "activity_data.csv")
    warmup_voting_data.to_csv(warmup_dir / "voting_data.csv")
    
    # Save population summary
    save_population_summary(
        warmup_state['agents'],
        warmup_dir / "population_summary.txt"
    )
    
    # Save warmup state for later use in other simulations
    print(f"Saving warmup state to {warmup_dir / 'warmup_state.pkl'}")
    pickle_start = time.time()
    with open(warmup_dir / "warmup_state.pkl", 'wb') as f:
        pickle.dump(warmup_state, f)
    print(f"Warmup state saved in {time.time() - pickle_start:.2f} seconds")
    log_memory_usage("After saving warmup state")
    
    print("\nWarmup stage completed!")
    print(f"Results saved to {warmup_dir}")
    print(f"Population summary saved to {warmup_dir}/population_summary.txt")
    print(f"Warmup state saved to {warmup_dir}/warmup_state.pkl")

if __name__ == "__main__":
    main() 