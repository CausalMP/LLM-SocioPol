#!/usr/bin/env python
"""
Runner script for main stages using the saved warmup state.
Implements the different treatment settings for the main stages.
"""
import argparse
import pickle
import os
import psutil
import time
from pathlib import Path
from src.simulation.simulation import run_simulation, log_memory_usage
from src.simulation.simulation_utils import save_user_interactions, save_network_data, calculate_and_print_turnout
from src.agents.persona import save_all_personas
from src.config import treatment_settings, time_points, random_seed, treatment_seed, n_runs

def main():
    parser = argparse.ArgumentParser(description='Run main simulation stages using the saved warmup state.')
    parser.add_argument('--n_users', type=int, default=3, help='Number of users')
    parser.add_argument('--topic', type=str, default='Politics', help='Topic string')
    parser.add_argument('--num_contents', type=int, default=20, help='Number of content pieces')
    parser.add_argument('--feed_length', type=int, default=4, help='Feed length')
    parser.add_argument('--warmup_dir', type=str, default='simulation_results/warmup', help='Directory with warmup state')
    parser.add_argument('--output_dir', type=str, default='simulation_results', help='Base directory to save outputs')
    parser.add_argument('--n_cores', type=int, default=8, help='Number of CPU cores to use for parallel processing')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of users to process in one batch')
    args = parser.parse_args()
    
    # Print system information
    print(f"Total system memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load warmup state
    warmup_state_file = Path(args.warmup_dir) / "warmup_state.pkl"
    if not warmup_state_file.exists():
        raise FileNotFoundError(f"Warmup state file not found at {warmup_state_file}. Run run_warmup.py first.")
    
    print(f"Loading warmup state from {warmup_state_file}")
    start_time = time.time()
    with open(warmup_state_file, 'rb') as f:
        warmup_state = pickle.load(f)
    print(f"Warmup state loaded in {time.time() - start_time:.2f} seconds")
    log_memory_usage("After loading warmup state")
    
    # Read warmup data files for later merging
    warmup_dir = Path(args.warmup_dir)
    warmup_engagement_data = None
    warmup_treatment_data = None
    warmup_activity_data = None
    warmup_voting_data = None
    
    try:
        import pandas as pd
        warmup_engagement_data = pd.read_csv(warmup_dir / "engagement_data.csv", index_col=0)
        warmup_treatment_data = pd.read_csv(warmup_dir / "treatment_data.csv", index_col=0)
        warmup_activity_data = pd.read_csv(warmup_dir / "activity_data.csv", index_col=0)
        warmup_voting_data = pd.read_csv(warmup_dir / "voting_data.csv", index_col=0)
    except FileNotFoundError as e:
        print(f"Warning: Could not load all warmup data files: {e}")
    
    # Set election day to the maximum time point
    election_day = max(time_points)
    
    # Calculate appropriate batch size based on user count
    recommended_batch_size = min(args.batch_size, max(100, args.n_users // 20))
    print(f"Using batch size of {recommended_batch_size} users")
    
    # Separate experiment settings from non-experiment settings
    experiment_settings = []
    non_experiment_settings = []
    
    for setting in treatment_settings:
        if "experiment" in setting[1]:
            experiment_settings.append(setting)
        else:
            non_experiment_settings.append(setting)
    
    print(f"\n===== RUNNING NON-EXPERIMENT SETTINGS =====")
    
    # Run non-experiment settings only once
    for treatment_probs, setting_name in non_experiment_settings:
        print(f"\nRunning setting: {setting_name}")
        
        # Ensure treatment_probs has the right length
        if len(treatment_probs) < len(time_points):
            treatment_probs.extend([treatment_probs[-1]] * (len(time_points) - len(treatment_probs)))
        
        # Create directory for this setting
        setting_dir = Path(args.output_dir) / setting_name
        setting_dir.mkdir(exist_ok=True)
        
        # Run simulation with warmup state
        engagement_data, treatment_data, activity_data, voting_data, final_state = run_simulation(
            n_users=args.n_users,
            time_points=time_points,
            treatment_probs=treatment_probs,
            topic=args.topic,
            c=args.num_contents,
            feed_length=args.feed_length,
            election_day=election_day,
            random_seed=random_seed,
            treatment_seed=treatment_seed,  # Use the default treatment seed
            initial_state=warmup_state,
            output_dir=str(setting_dir),
            n_cores=args.n_cores,
            setting_name=setting_name,
            batch_size=recommended_batch_size
        )
        
        # Combine with warmup data if available
        if warmup_engagement_data is not None:
            # Ensure indices are compatible
            combined_engagement_data = pd.concat([warmup_engagement_data, engagement_data], axis=1)
            combined_treatment_data = pd.concat([warmup_treatment_data, treatment_data], axis=1)
            combined_activity_data = pd.concat([warmup_activity_data, activity_data], axis=1)
            combined_voting_data = pd.concat([warmup_voting_data, voting_data], axis=1)
            
            # Save combined data
            combined_engagement_data.to_csv(setting_dir / f"{setting_name}_engagement_data.csv")
            combined_treatment_data.to_csv(setting_dir / f"{setting_name}_treatment_data.csv")
            combined_activity_data.to_csv(setting_dir / f"{setting_name}_activity_data.csv")
            combined_voting_data.to_csv(setting_dir / f"{setting_name}_voting_data.csv")
        
        # Save final state for this setting
        (setting_dir / "more_results").mkdir(exist_ok=True)
        with open(setting_dir / "more_results" / f"{setting_name}_final_state.pkl", 'wb') as f:
            pickle.dump(final_state, f)
        
        # Save users' interactions
        for user_id in final_state['agents']:
            save_user_interactions(
                user_id,
                final_state['agents'],
                final_state['contents'],
                setting_dir / "more_results" / f"user_{user_id}_interactions.txt"
            )
        
        # Save all personas for this treatment setting
        save_all_personas(
            final_state['agents'],
            setting_dir / "more_results" / f"{setting_name}_personas.csv"
        )
        
        # Save network data
        save_network_data(
            final_state['agents'],
            setting_dir / "more_results" / f"{setting_name}_network.csv"
        )
        
        # Calculate and print election turnout
        calculate_and_print_turnout(final_state['agents'], setting_name)
        
        print(f"Setting {setting_name} completed. Results saved to {setting_dir}")
        
        # Clear memory
        import gc
        del final_state, engagement_data, treatment_data, activity_data, voting_data
        gc.collect()
        log_memory_usage(f"After completing setting {setting_name}")
    
    # Run experiment settings multiple times with different treatment seeds
    if experiment_settings:
        print(f"\n===== RUNNING EXPERIMENT SETTINGS WITH {n_runs} DIFFERENT TREATMENT SEEDS =====")
        for run_idx in range(n_runs):
            # Generate a different treatment seed for each run
            run_treatment_seed = treatment_seed + run_idx
            
            print(f"\n===== EXPERIMENT RUN {run_idx+1}/{n_runs} WITH TREATMENT SEED {run_treatment_seed} =====")
            
            # Run each experiment setting
            for treatment_probs, setting_name in experiment_settings:
                print(f"\nRunning setting: {setting_name}")
                
                # Ensure treatment_probs has the right length
                if len(treatment_probs) < len(time_points):
                    treatment_probs.extend([treatment_probs[-1]] * (len(time_points) - len(treatment_probs)))
                
                # Create directory for this setting under the treatment seed
                setting_dir = Path(args.output_dir) / f"treatment_seed_{run_treatment_seed}" / setting_name
                setting_dir.mkdir(exist_ok=True, parents=True)
                
                # Run simulation with warmup state
                engagement_data, treatment_data, activity_data, voting_data, final_state = run_simulation(
                    n_users=args.n_users,
                    time_points=time_points,
                    treatment_probs=treatment_probs,
                    topic=args.topic,
                    c=args.num_contents,
                    feed_length=args.feed_length,
                    election_day=election_day,
                    random_seed=random_seed,
                    treatment_seed=run_treatment_seed,  # Use the varied treatment seed
                    initial_state=warmup_state,
                    output_dir=str(setting_dir),
                    n_cores=args.n_cores,
                    setting_name=setting_name,
                    batch_size=recommended_batch_size
                )
                
                # Combine with warmup data if available
                if warmup_engagement_data is not None:
                    # Ensure indices are compatible
                    combined_engagement_data = pd.concat([warmup_engagement_data, engagement_data], axis=1)
                    combined_treatment_data = pd.concat([warmup_treatment_data, treatment_data], axis=1)
                    combined_activity_data = pd.concat([warmup_activity_data, activity_data], axis=1)
                    combined_voting_data = pd.concat([warmup_voting_data, voting_data], axis=1)
                    
                    # Save combined data
                    combined_engagement_data.to_csv(setting_dir / f"{setting_name}_engagement_data.csv")
                    combined_treatment_data.to_csv(setting_dir / f"{setting_name}_treatment_data.csv")
                    combined_activity_data.to_csv(setting_dir / f"{setting_name}_activity_data.csv")
                    combined_voting_data.to_csv(setting_dir / f"{setting_name}_voting_data.csv")
                
                # Save final state for this setting
                (setting_dir / "more_results").mkdir(exist_ok=True)
                with open(setting_dir / "more_results" / f"{setting_name}_final_state.pkl", 'wb') as f:
                    pickle.dump(final_state, f)
                
                # Save users' interactions
                for user_id in final_state['agents']:
                    save_user_interactions(
                        user_id,
                        final_state['agents'],
                        final_state['contents'],
                        setting_dir / "more_results" / f"user_{user_id}_interactions.txt"
                    )
                
                # Save all personas for this treatment setting
                save_all_personas(
                    final_state['agents'],
                    setting_dir / "more_results" / f"{setting_name}_personas.csv"
                )
                
                # Save network data
                save_network_data(
                    final_state['agents'],
                    setting_dir / "more_results" / f"{setting_name}_network.csv"
                )
                
                # Calculate and print election turnout
                calculate_and_print_turnout(final_state['agents'], setting_name)
                
                print(f"Setting {setting_name} with treatment seed {run_treatment_seed} completed. Results saved to {setting_dir}")
                
                # Clear memory
                import gc
                del final_state, engagement_data, treatment_data, activity_data, voting_data
                gc.collect()
                log_memory_usage(f"After completing experiment setting {setting_name} with seed {run_treatment_seed}")
            
            print(f"\nExperiment run {run_idx+1} completed with treatment seed {run_treatment_seed}")
    
    print("\nAll simulations completed!")

if __name__ == "__main__":
    main() 