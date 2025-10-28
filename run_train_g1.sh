#!/bin/bash

# Training script for G1 Diffusion Policy
# This script ensures the correct conda environment is activated
# Usage: ./run_train_g1.sh [--config baseline|improved]

set -e  # Exit on error

echo "=========================================="
echo "G1 Diffusion Policy Training"
echo "=========================================="
echo ""

# Navigate to the correct directory
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot

# Activate conda environment
echo "Activating conda environment: unitree_IL"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate unitree_IL

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run training (pass through all arguments)
echo "Starting training..."
echo ""
python train_g1_diffusion.py "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

