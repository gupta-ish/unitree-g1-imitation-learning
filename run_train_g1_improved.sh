#!/bin/bash

# Training script for G1 Diffusion Policy - IMPROVED Configuration
# This script ensures the correct conda environment is activated
# Uses improved hyperparameters for better performance

set -e  # Exit on error

echo "=========================================="
echo "G1 Diffusion Policy Training - IMPROVED"
echo "=========================================="
echo ""
echo "Configuration: improved"
echo "  - Training steps: 250,000"
echo "  - Learning rate: 5e-5"
echo "  - Batch size: 16"
echo "  - Weight decay: 5e-6"
echo "  - Gradient clip: 5.0"
echo "  - Warmup steps: 1,000"
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

# Run training with improved config
echo "Starting training with improved configuration..."
echo ""
python train_g1_diffusion.py --config improved

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="


