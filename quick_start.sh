#!/bin/bash

# Quick start script for DQN training

echo "=================================="
echo "DQN Car Racing - Quick Start"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run tests
echo ""
echo "Running component tests..."
cd src
python test_components.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Setup complete!"
    echo "=================================="
    echo ""
    echo "To start training:"
    echo "  cd src"
    echo "  python train.py --config ../configs/dqn_config.yaml"
    echo ""
    echo "To monitor training:"
    echo "  tensorboard --logdir ../results/logs"
    echo ""
    echo "To evaluate a trained model:"
    echo "  python evaluate.py --model ../models/dqn/best_model.pth --episodes 10 --render"
    echo ""
else
    echo ""
    echo "Setup failed. Please check the errors above."
    exit 1
fi
