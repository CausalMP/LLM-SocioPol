#!/bin/bash

### Example command:
# Only run the warmup stage
# bash run_all_stages.sh --n_users 100 --skip_main

# Skip warmup if already done
# bash run_all_stages.sh --n_users 100 --skip_warmup

# Run with batch processing for large user counts
# bash run_all_stages.sh --n_users 20000 --n_cores 100 --batch_size 500



# run_all_stages.sh
# Script to run warmup and all simulation stages in sequence

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load API key from file
if [ -f "$DIR/OPENAI_API_KEY.env" ]; then
    # Extract just the key value (without export, quotes, etc.)
    OPENAI_API_KEY=$(grep -o 'OPENAI_API_KEY=[^"]*' "$DIR/OPENAI_API_KEY.env" | cut -d'=' -f2)
    export OPENAI_API_KEY
    echo "API key loaded from OPENAI_API_KEY.env"
    echo "Key (truncated): ${OPENAI_API_KEY:0:10}...${OPENAI_API_KEY: -5}"
else
    echo "OPENAI_API_KEY.env file not found"
    exit 1
fi

# Check for required Python packages
check_dependencies() {
  echo "Checking for required Python packages..."
  if ! $PYTHON_CMD -c "import psutil" &> /dev/null || ! $PYTHON_CMD -c "import openai" &> /dev/null; then
    echo "Required packages missing. Installing dependencies..."
    bash install_dependencies.sh
    if [ $? -ne 0 ]; then
      echo "Error: Failed to install dependencies. Please install them manually:"
      echo "$PYTHON_CMD -m pip install --user -r requirements.txt"
      return 1
    fi
    echo "Dependencies installed successfully."
  else
    echo "All required packages are installed."
  fi
  return 0
}

# Show help and exit if --help flag is present
if [[ "$*" == *"--help"* ]]; then
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --n_users N            Number of users (default: 3)"
  echo "  --topic STR            Topic string (default: 'Politics')"
  echo "  --num_contents N       Number of content pieces (default: 20)"
  echo "  --feed_length N        Feed length (default: 4)"
  echo "  --output_dir DIR       Output directory (default: 'simulation_results')"
  echo "  --n_cores N            Number of CPU cores (default: 8)"
  echo "  --batch_size N         Number of users to process in one batch (default: 500)"
  echo "  --skip_warmup          Skip warmup stage if already run"
  echo "  --skip_main            Skip main stages"
  exit 0
fi

# Set default values
N_USERS=3
TOPIC="Politics"
NUM_CONTENTS=20
FEED_LENGTH=4
N_CORES=8
BATCH_SIZE=500
OUTPUT_DIR="simulation_results"
SKIP_WARMUP=false
SKIP_MAIN=false
N_RUNS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --n_users)
      N_USERS="$2"
      shift 2
      ;;
    --topic)
      TOPIC="$2"
      shift 2
      ;;
    --num_contents)
      NUM_CONTENTS="$2"
      shift 2
      ;;
    --feed_length)
      FEED_LENGTH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --n_cores)
      N_CORES="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --skip_warmup)
      SKIP_WARMUP=true
      shift
      ;;
    --skip_main)
      SKIP_MAIN=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "Running simulation with the following parameters:"
echo "  Number of users: $N_USERS"
echo "  Topic: $TOPIC"
echo "  Number of contents: $NUM_CONTENTS"
echo "  Feed length: $FEED_LENGTH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of CPU cores: $N_CORES"
echo "  Batch size: $BATCH_SIZE"

# Calculate recommended batch size for large simulations
if [ "$N_USERS" -gt 1000 ]; then
  # More intelligent batch size calculation based on user count and cores
  # For large user counts, aim for ~20-25 batches spread across cores
  RECOMMENDED_BATCH_SIZE=$(( N_USERS / (N_CORES / 4) ))
  
  # Cap at reasonable values
  if [ "$RECOMMENDED_BATCH_SIZE" -gt 500 ]; then
    RECOMMENDED_BATCH_SIZE=500
  elif [ "$RECOMMENDED_BATCH_SIZE" -lt 100 ]; then
    RECOMMENDED_BATCH_SIZE=100
  fi
  
  # If user provided a batch size that's significantly larger, warn them
  if [ "$BATCH_SIZE" -gt $(( RECOMMENDED_BATCH_SIZE * 2 )) ]; then
    echo "Warning: Your batch size ($BATCH_SIZE) is much larger than recommended ($RECOMMENDED_BATCH_SIZE) for $N_USERS users on $N_CORES cores"
    echo "Consider using --batch_size $RECOMMENDED_BATCH_SIZE to prevent memory issues"
  # If user provided a batch size that's significantly smaller, warn about performance
  elif [ "$BATCH_SIZE" -lt $(( RECOMMENDED_BATCH_SIZE / 2 )) ] && [ "$BATCH_SIZE" -lt 100 ]; then
    echo "Note: Your batch size ($BATCH_SIZE) is much smaller than recommended ($RECOMMENDED_BATCH_SIZE)"
    echo "This may lead to slower performance but could help with memory constraints"
  fi
fi

# Create log directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WARMUP_LOG="$LOG_DIR/warmup_${TIMESTAMP}.log"
MAIN_LOG="$LOG_DIR/main_${TIMESTAMP}.log"

# Ensure the warmup directory exists
WARMUP_DIR="$OUTPUT_DIR/warmup"

# Run warmup stage
if [ "$SKIP_WARMUP" = false ]; then
  echo -e "\n===== RUNNING WARMUP STAGE ====="
  echo "Logging to $WARMUP_LOG"
  
  # Print system info
  echo "System Information:" | tee -a "$WARMUP_LOG"
  if command -v free &> /dev/null; then
    echo "Total Memory: $(free -h | grep Mem | awk '{print $2}')" | tee -a "$WARMUP_LOG"
    echo "Available Memory: $(free -h | grep Mem | awk '{print $4}')" | tee -a "$WARMUP_LOG"
  else
    # macOS alternative
    echo "Total Memory: $(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}')" | tee -a "$WARMUP_LOG"
  fi
  echo "CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc)" | tee -a "$WARMUP_LOG"
  
  # Check which Python is available
  echo "Checking for Python..." | tee -a "$WARMUP_LOG"
  if command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "Using $(which python) - version $(python --version 2>&1)" | tee -a "$WARMUP_LOG"
  elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Using $(which python3) - version $(python3 --version 2>&1)" | tee -a "$WARMUP_LOG"
  else
    echo "ERROR: No Python found! Please load a Python module or environment." | tee -a "$WARMUP_LOG"
    exit 1
  fi
  
  # Check for dependencies
  check_dependencies
  if [ $? -ne 0 ]; then
    echo "Error: Missing dependencies. Please install them manually." | tee -a "$WARMUP_LOG"
    exit 1
  fi
  
  $PYTHON_CMD run_warmup.py \
    --n_users "$N_USERS" \
    --topic "$TOPIC" \
    --num_contents "$NUM_CONTENTS" \
    --feed_length "$FEED_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --n_cores "$N_CORES" \
    --batch_size "$BATCH_SIZE" 2>&1 | tee -a "$WARMUP_LOG"
    
  # Check if warmup completed successfully
  if [ $? -ne 0 ]; then
    echo "Error: Warmup stage failed. Check $WARMUP_LOG for details." | tee -a "$WARMUP_LOG"
    exit 1
  fi
else
  echo "Skipping warmup stage..."
  
  # Check if warmup state exists
  if [ ! -f "$WARMUP_DIR/warmup_state.pkl" ]; then
    echo "Error: Warmup state not found at $WARMUP_DIR/warmup_state.pkl"
    echo "Please run warmup stage first or remove --skip_warmup flag."
    exit 1
  fi
fi

# Run main stages
if [ "$SKIP_MAIN" = false ]; then
  echo -e "\n===== RUNNING MAIN STAGES ====="
  echo "Logging to $MAIN_LOG"
  
  # Check which Python is available
  echo "Checking for Python..." | tee -a "$MAIN_LOG"
  if command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "Using $(which python) - version $(python --version 2>&1)" | tee -a "$MAIN_LOG"
  elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Using $(which python3) - version $(python3 --version 2>&1)" | tee -a "$MAIN_LOG"
  else
    echo "ERROR: No Python found! Please load a Python module or environment." | tee -a "$MAIN_LOG"
    exit 1
  fi
  
  # Check for dependencies again for main stage
  check_dependencies
  if [ $? -ne 0 ]; then
    echo "Error: Missing dependencies. Please install them manually." | tee -a "$MAIN_LOG"
    exit 1
  fi
  
  $PYTHON_CMD run_main_stages.py \
    --n_users "$N_USERS" \
    --topic "$TOPIC" \
    --num_contents "$NUM_CONTENTS" \
    --feed_length "$FEED_LENGTH" \
    --warmup_dir "$WARMUP_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_cores "$N_CORES" \
    --batch_size "$BATCH_SIZE" 2>&1 | tee -a "$MAIN_LOG"
    
  # Check if main stages completed successfully
  if [ $? -ne 0 ]; then
    echo "Error: Main stages failed. Check $MAIN_LOG for details." | tee -a "$MAIN_LOG"
    exit 1
  fi
else
  echo "Skipping main stages..."
fi

echo -e "\n===== ALL SIMULATIONS COMPLETED ====="
echo "Results saved to $OUTPUT_DIR"
echo "Logs saved to $LOG_DIR" 