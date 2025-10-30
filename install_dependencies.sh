#!/bin/bash
# Script to install required Python packages

echo "Installing required Python packages..."

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "ERROR: No Python found! Please load a Python module or environment."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version 2>&1)"
echo "Python location: $(which $PYTHON_CMD)"

# Install pip if not available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py --user
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install pip. Please install pip manually."
        exit 1
    fi
    rm get-pip.py
fi

# Make sure requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

# Display content of requirements.txt
echo "Contents of requirements.txt:"
cat requirements.txt

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
$PYTHON_CMD -m pip install --user -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies. Please check the error messages above."
    exit 1
fi

# Verify all required packages are installed
echo "Verifying installation of required packages..."
MISSING_PACKAGES=0

check_package() {
    echo -n "Checking for $1... "
    if $PYTHON_CMD -c "import $1" &> /dev/null; then
        echo "OK"
        return 0
    else
        echo "MISSING"
        return 1
    fi
}

# Check each required package
check_package "psutil" || MISSING_PACKAGES=$((MISSING_PACKAGES+1))
check_package "numpy" || MISSING_PACKAGES=$((MISSING_PACKAGES+1))
check_package "pandas" || MISSING_PACKAGES=$((MISSING_PACKAGES+1))
check_package "matplotlib" || MISSING_PACKAGES=$((MISSING_PACKAGES+1))
check_package "tqdm" || MISSING_PACKAGES=$((MISSING_PACKAGES+1))
check_package "openai" || MISSING_PACKAGES=$((MISSING_PACKAGES+1))

# Note: pickle5 is only needed for Python < 3.8, Python 3.8+ has pickle5 functionality built-in
if [[ $($PYTHON_CMD --version 2>&1) == *"3.7"* ]] || [[ $($PYTHON_CMD --version 2>&1) == *"3.6"* ]]; then
    check_package "pickle5" || MISSING_PACKAGES=$((MISSING_PACKAGES+1))
fi

if [ $MISSING_PACKAGES -gt 0 ]; then
    echo "WARNING: $MISSING_PACKAGES package(s) could not be imported."
    echo "Try installing them manually with:"
    echo "$PYTHON_CMD -m pip install --user --no-cache-dir -r requirements.txt"
    exit 1
else
    echo "All dependencies installed successfully!"
fi 