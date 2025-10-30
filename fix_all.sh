#!/bin/bash
# fix_all.sh - A comprehensive script to fix all dependencies and environment issues

echo "===== Comprehensive Fix Script for LLM Election Simulation ====="

# Find Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Using $(which python3) - version $($PYTHON_CMD --version 2>&1)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "Using $(which python) - version $($PYTHON_CMD --version 2>&1)"
else
    echo "ERROR: No Python found! Attempting to load Python module..."
    module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "Using $(which python3) - version $($PYTHON_CMD --version 2>&1)"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        echo "Using $(which python) - version $($PYTHON_CMD --version 2>&1)"
    else
        echo "CRITICAL ERROR: No Python found. Please load a Python module or environment."
        exit 1
    fi
fi

# Get user site-packages directory
USER_SITE=$($PYTHON_CMD -m site --user-site)
USER_BASE=$($PYTHON_CMD -m site --user-base)
echo "User site-packages directory: $USER_SITE"
echo "User base directory: $USER_BASE"

# Ensure user site-packages directory exists
mkdir -p "$USER_SITE"

# Set environment variables
export PYTHONPATH="$USER_SITE:$PYTHONPATH"
export PYTHONUSERBASE="$HOME/.local"
echo "Set PYTHONPATH=$PYTHONPATH"
echo "Set PYTHONUSERBASE=$PYTHONUSERBASE"

# Create a diagnostic file
DIAGNOSTIC_FILE="python_env_diagnostic.txt"
echo "Creating diagnostic file: $DIAGNOSTIC_FILE"

$PYTHON_CMD -c "
import sys, os, site
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Python installation prefix:', sys.prefix)
print('\\nPython path:')
for p in sys.path:
    print(f'  {p}')
print('\\nUser site-packages:', site.USER_SITE)
print('User base directory:', site.USER_BASE)
print('\\nEnvironment variables:')
for k, v in os.environ.items():
    if 'PYTHON' in k or 'PATH' in k:
        print(f'  {k}={v}')
" > "$DIAGNOSTIC_FILE"

echo "Diagnostic information saved to $DIAGNOSTIC_FILE"

# Step 1: Fix NumPy version if needed
echo "===== Step 1: Checking NumPy/SciPy Compatibility ====="
NUMPY_VERSION=$($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
echo "Current NumPy version: $NUMPY_VERSION"

if [[ "$NUMPY_VERSION" == 2* ]]; then
    echo "NumPy version 2.x detected. This is incompatible with SciPy."
    echo "Downgrading NumPy to a compatible version..."
    $PYTHON_CMD -m pip uninstall -y numpy
    $PYTHON_CMD -m pip install --user --no-cache-dir "numpy>=1.17.3,<1.25.0"
    
    # Verify NumPy installation
    NEW_NUMPY_VERSION=$($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
    echo "New NumPy version: $NEW_NUMPY_VERSION"
fi

# Step 2: Install all required packages
echo "===== Step 2: Installing All Required Packages ====="
REQUIREMENTS=(
    "psutil>=5.9.0"
    "numpy>=1.17.3,<1.25.0"  # Compatible with SciPy
    "pandas>=1.3.0"
    "pickle5>=0.0.11"
    "matplotlib>=3.5.0"
    "tqdm>=4.62.0"
    "openai>=1.0.0"
    "anthropic>=0.5.0"
    "scipy>=1.7.0"
    "transformers>=4.20.0"
    "torch>=1.10.0"
    "regex>=2022.3.15"
    "typing-extensions>=4.0.0"
    "json5>=0.9.6"
)

for pkg in "${REQUIREMENTS[@]}"; do
    echo "Installing $pkg..."
    $PYTHON_CMD -m pip install --user --no-cache-dir "$pkg"
done

# Step 3: Verify all installations
echo "===== Step 3: Verifying All Installations ====="
PACKAGES=("psutil" "openai" "numpy" "pandas" "matplotlib" "tqdm" "anthropic" "transformers" "torch" "regex" "typing_extensions" "json5" "scipy")
MISSING=0

for pkg in "${PACKAGES[@]}"; do
    echo -n "Checking $pkg... "
    if $PYTHON_CMD -c "import $pkg" &> /dev/null; then
        version=$($PYTHON_CMD -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
        echo "OK (version $version)"
    else
        echo "MISSING"
        MISSING=$((MISSING+1))
    fi
done

# Step 4: Special checks for critical functionality
echo "===== Step 4: Testing Critical Functionality ====="

# Test SciPy functionality
echo "Testing SciPy optimization module..."
if $PYTHON_CMD -c "from scipy.optimize import linear_sum_assignment; print('SciPy optimization test: OK')" 2>/dev/null; then
    echo "SciPy optimization module is working correctly!"
else
    echo "SciPy optimization module has issues. See error above."
    MISSING=$((MISSING+1))
fi

# Test transformers functionality
echo "Testing transformers module..."
if $PYTHON_CMD -c "from transformers import AutoTokenizer; print('Transformers test: OK')" 2>/dev/null; then
    echo "Transformers module is working correctly!"
else
    echo "Transformers module has issues. See error above."
    MISSING=$((MISSING+1))
fi

# Step 5: Create helper scripts
echo "===== Step 5: Creating Helper Scripts ====="

# Create a wrapper script to use the correct Python environment
echo "Creating Python wrapper script..."
cat > python_wrapper.sh << EOF
#!/bin/bash
# Python wrapper script with proper environment
export PYTHONPATH="$USER_SITE:\$PYTHONPATH"
export PYTHONUSERBASE="$HOME/.local"
$PYTHON_CMD "\$@"
EOF
chmod +x python_wrapper.sh
echo "Created python_wrapper.sh - use this to run Python with proper environment"

# Create environment setup script
echo "Creating environment setup script..."
cat > setup_python_env.sh << EOF
#!/bin/bash
# Source this file to set up Python environment
export PYTHONPATH="$USER_SITE:\$PYTHONPATH"
export PYTHONUSERBASE="$HOME/.local"
echo "Python environment set up with PYTHONPATH=$USER_SITE:\$PYTHONPATH"
EOF
chmod +x setup_python_env.sh
echo "Created setup_python_env.sh - source this file before running Python scripts"

# Final result
if [ $MISSING -gt 0 ]; then
    echo "===== FIX INCOMPLETE ====="
    echo "There were $MISSING issues that could not be fixed automatically."
    echo "Please check the output above for error messages."
    echo "You may need to fix some issues manually."
    exit 1
else
    echo "===== FIX COMPLETED SUCCESSFULLY ====="
    echo "All dependencies have been installed and verified."
    echo "All critical functionality is working correctly."
    echo ""
    echo "To run your simulation:"
    echo "1. Source the environment: source setup_python_env.sh"
    echo "2. Submit the job: sbatch run_election_job.sh"
    exit 0
fi 