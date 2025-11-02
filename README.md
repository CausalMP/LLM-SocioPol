# LLM Social-Political Mobilization

An open-source Python software for simulating LLM-based social networks to study causal effects of social influence on voting behavior, including data from five independent experimental runs.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{shirani2025simulating,
  title={Simulating and Experimenting with Social Media Mobilization Using LLM Agents},
  author={Shirani, Sadegh and Bayati, Mohsen},
  journal={arXiv preprint arXiv:2510.26494},
  year={2025}
}
```

## Overview

The simulation creates a social network of LLM agents (personas) who interact with content and each other. The experiment measures how different interventions (treatment messages) affect voting behavior.

## Dependencies

The simulation requires several Python packages. You can install them using the provided installation script:

```bash
# Install dependencies
chmod +x install_dependencies.sh
./install_dependencies.sh
```

Required packages:
- psutil
- numpy
- pandas
- pickle5
- matplotlib
- tqdm
- openai

## Running the Simulation

The simulation is designed to be run through a single script:

```bash
bash run_all_stages.sh [options]
```

### Example Commands

```bash
# Run a full simulation with 100 users
bash run_all_stages.sh --n_users 100

# Run only the warmup stage with 100 users
bash run_all_stages.sh --n_users 100 --skip_main

# Skip warmup if already done and run only main stages
bash run_all_stages.sh --n_users 100 --skip_warmup

# Run a large-scale simulation with batch processing and more cores
bash run_all_stages.sh --n_users 20000 --n_cores 100 --batch_size 500
```

### Command Line Options

- `--n_users N`: Number of users in the simulation (default: 3)
- `--topic STR`: Topic for the simulation (default: 'Politics')
- `--num_contents N`: Number of content pieces (default: 20)
- `--feed_length N`: Feed length (default: 4)
- `--output_dir DIR`: Output directory (default: 'simulation_results')
- `--n_cores N`: Number of CPU cores (default: 8)
- `--batch_size N`: Number of users to process in one batch (default: 1000)
- `--skip_warmup`: Skip warmup stage if already run
- `--skip_main`: Skip main stages

## Large-Scale Simulations

The simulation includes memory monitoring and batch processing capabilities for running large-scale experiments:

1. **Batch Processing**: Users are processed in batches to manage memory usage. For large simulations (e.g., 20,000 users), consider using a batch size of 500-1000.

2. **Memory Monitoring**: The system monitors memory usage throughout the simulation and adjusts batch sizes dynamically if memory becomes critical.

3. **Logging**: Detailed logs are saved to the `logs` directory within your output directory, containing memory usage statistics and processing times.

For large-scale simulations, recommended parameters are:

```bash
bash run_all_stages.sh --n_users 20000 --n_cores 100 --batch_size 500 --output_dir simulation_results_large
```

## API Key Setup

The simulation requires an OpenAI API key. Follow these steps:

1. Copy the example environment file:
   ```bash
   cp OPENAI_API_KEY.env.example OPENAI_API_KEY.env
   ```

2. Edit `OPENAI_API_KEY.env` and replace `your_openai_api_key_here` with your actual OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

⚠️ **Important**: Never commit your actual API key to version control. The `OPENAI_API_KEY.env` file is already included in `.gitignore`.

## Troubleshooting

If you encounter dependency issues, especially in HPC environments, the `fix_all.sh` script provides comprehensive dependency management and environment setup. This script is designed for complex environments and may not be needed for standard installations.

## Included Experimental Data

This repository includes pre-run experimental results from 5 independent simulation runs with different random seeds:

- **Location**: `result_data/seed_1/` through `result_data/seed_5/`
- **Experimental Conditions**: Each seed includes results for:
  - `control` - No treatment messages
  - `treatment_info_message` - Full informational message treatment (100% probability)
  - `treatment_soc_message` - Full social message treatment (100% probability)
  - `experiment_info_message` - Gradual informational message treatment (20%, 40%, 80%, 80%)
  - `experiment_soc_message` - Gradual social message treatment (10%, 20%, 50%, 50%)
- **Data Files**: Each condition contains:
  - `*_personas.csv` - User persona information and voting turnout
  - `*_treatment_data.csv` - Treatment application records
  - `*_voting_data.csv` - Voting behavior outcomes

These results can be used for analysis without needing to re-run the simulations, which can be computationally expensive.

## Project Structure

The codebase is organized as follows:

```
.
├── src/                     # Main source code directory
│   ├── agents/              # Agent-related code
│   │   ├── agent.py         # Agent class implementation
│   │   └── persona.py       # Persona generation and management
│   ├── models/              # Model-related code
│   │   └── feed_ranking.py  # Feed ranking algorithms
│   ├── simulation/          # Simulation-related code
│   │   ├── simulation.py    # Main simulation functions
│   │   ├── simulation_utils.py  # Utility functions for simulation
│   │   └── run_multiple.py  # Functions for running multiple simulations
│   ├── utils/               # Utility functions
│   │   └── llm_utils.py     # LLM interaction utilities
│   └── config.py            # Global configuration parameters
├── result_data/             # Pre-run experimental results (5 seeds)
├── users_data/              # User profiles and network data
├── run_all_stages.sh        # Main script to run the complete simulation
├── run_warmup.py            # Script for the warmup stage
├── run_main_stages.py       # Script for the main stages
├── OPENAI_API_KEY.env       # OpenAI API key
└── README.md                # This file
```

## How the Simulation Works

The simulation runs in two main stages:

1. **Warmup Stage** (`run_warmup.py`): Initializes user personas, relationship networks, and runs several rounds of interaction to establish baseline behaviors.

2. **Main Stages** (`run_main_stages.py`): Runs the main experiment with different treatment settings as defined in `src/config.py`.

The simulation creates several types of output data:
- Engagement data: How users interact with content
- Treatment data: Which treatments were applied to users
- Activity data: User activities over time
- Voting data: User voting intentions
- User interactions: Detailed logs of user interactions
- Network data: Social network information
- Persona data: User persona information

Results are saved to the specified output directory (default: `simulation_results`). 