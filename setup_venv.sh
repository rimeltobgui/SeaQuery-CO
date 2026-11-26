#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TARGET_VERSION="3.12.4"
VENV_DIR="${1:-.seaquery_co_venv}"

# 1. Check if the exact Python 3.12.4 version is already available
if command -v python3.12 &> /dev/null && [[ "$(python3.12 --version)" == *"3.12.4"* ]]; then
    PYTHON_BIN=$(command -v python3.12)
elif [ -x "$HOME/.pyenv/versions/$TARGET_VERSION/bin/python" ]; then
    PYTHON_BIN="$HOME/.pyenv/versions/$TARGET_VERSION/bin/python"
else
    echo "Python $TARGET_VERSION not found. Installing via pyenv..."
    
    # Install build dependencies required to compile Python on Ubuntu
    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev

    # Install pyenv if not already installed
    if [ ! -d "$HOME/.pyenv" ]; then
        curl https://pyenv.run | bash
    fi

    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    # Install exact Python 3.12.4
    pyenv install "$TARGET_VERSION"
    PYTHON_BIN="$HOME/.pyenv/versions/$TARGET_VERSION/bin/python"
fi

# 2. Create the virtual environment using Python 3.12.4
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment '$VENV_DIR' with $($PYTHON_BIN --version)..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 3. Activate and update pip
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "Installing dependencies from $PROJECT_DIR/code/requirements.txt..."
python -m pip install -r "$PROJECT_DIR/code/requirements.txt"
python -m pip check

echo "Done! Active environment: $(python --version) inside '$VENV_DIR'"

# 4. Prevent Git from tracking the virtual environment
if ! grep -qxF "$VENV_DIR/" .gitignore 2>/dev/null; then
    echo "$VENV_DIR/" >> .gitignore
    echo "Added '$VENV_DIR/' to .gitignore"
fi

