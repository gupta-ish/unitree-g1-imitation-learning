#!/bin/bash
# Training script for ACT (Action Chunking Transformers) Policy on Unitree G1 Dataset
#
# Usage:
#   ./run_train_g1_act.sh [baseline|large]
#
# Examples:
#   ./run_train_g1_act.sh                # Run baseline config (default)
#   ./run_train_g1_act.sh large          # Run large config

# Get config from argument (default to baseline)
CONFIG=${1:-baseline}

echo "=========================================="
echo "ACT Policy Training - Unitree G1 Dataset"
echo "=========================================="
echo "Configuration: $CONFIG"
echo ""

# Activate conda environment
echo "Activating unitree_IL environment..."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate unitree_IL

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'unitree_IL'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
fi

# Verify Python and PyTorch
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

# Start training
echo "Starting ACT training with config: $CONFIG"
echo "Logs will be saved to: train_g1_act_${CONFIG}.log"
echo ""
echo "Press Ctrl+C to stop training"
echo "=========================================="
echo ""

python train_g1_act.py --config $CONFIG

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed or was interrupted"
    echo "=========================================="
    exit 1
fi

