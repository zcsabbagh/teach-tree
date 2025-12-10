#!/bin/bash

# Setup script for Inverse Cognitive Search experiment

set -e

echo "Setting up project with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    echo "Then run this script again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Check for .env.local file
if [ ! -f ".env.local" ]; then
    echo ""
    echo "WARNING: .env.local file not found!"
    echo "Please create .env.local with your Together API key:"
    echo "  TOGETHER_API_KEY=your_api_key_here"
    echo ""
else
    echo ".env.local found."
fi

echo ""
echo "Setup complete! To run the experiment:"
echo "  source .venv/bin/activate"
echo "  python diagnostic_search.py"
echo ""
echo "Or run directly with uv:"
echo "  uv run diagnostic_search.py"
