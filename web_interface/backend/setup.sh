#!/bin/bash
# Setup script for Abalone backend

echo "Setting up Abalone backend environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete! To run the backend:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run server: python main.py"