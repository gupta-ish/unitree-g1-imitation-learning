# Training Diffusion Policy on Unitree G1 Dataset

This document explains how to train a diffusion policy on your converted G1 dataset.

## Prerequisites

1. ✅ Dataset converted to LeRobot format (located at `/home/ishitagupta/zenavatar/lerobot_datasets/g1_dataset/`)
2. ✅ Conda environment `unitree_IL` with all dependencies installed
3. ✅ CUDA-enabled GPU (recommended)

## Quick Start

### Option 1: Using the Shell Script (Recommended)

**Baseline configuration (100k steps, faster):**
```bash
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot
./run_train_g1.sh --config baseline
```

**Improved configuration (250k steps, better performance):**
```bash
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot
./run_train_g1.sh --config improved
```

This will:
- Automatically activate the conda environment
- Verify CUDA availability
- Start training with the selected configuration

### Option 2: Direct Python Execution

```bash
conda activate unitree_IL
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot
python train_g1_diffusion.py --config baseline  # or --config improved
```

## Available Training Configurations

### Baseline Configuration
Quick training for testing and initial experiments (default if no config specified).

- **Training Steps:** 100,000 (~5-6 hours on RTX 4090)
- **Learning Rate:** 1e-4
- **Batch Size:** 8
- **Weight Decay:** 1e-6
- **Gradient Clip:** 10.0
- **Warmup Steps:** 500
- **Output Directory:** `outputs/train/g1_diffusion_baseline/`

### Improved Configuration (Recommended)
Extended training with optimized hyperparameters for best performance.

- **Training Steps:** 250,000 (~17-20 hours on RTX 4090)
- **Learning Rate:** 5e-5 (lower for finer optimization)
- **Batch Size:** 16 (larger for more stable gradients)
- **Weight Decay:** 5e-6 (higher for better generalization)
- **Gradient Clip:** 5.0 (tighter for stability)
- **Warmup Steps:** 1,000 (longer warmup)
- **Output Directory:** `outputs/train/g1_diffusion_improved/`

**Key Improvements:**
- 2.5x more training steps for better convergence
- Lower learning rate for finer-grained optimization
- Larger batch size for more stable gradient estimates
- Better regularization for improved generalization

## Checkpoint Structure

Each checkpoint includes:
- **Policy weights:** The trained model parameters
- **training_state.pt:** Optimizer and scheduler state for resuming training
- **hyperparameters.json:** Human-readable config used for this checkpoint

This makes it easy to see what hyperparameters were used for any checkpoint:
```bash
cat outputs/train/g1_diffusion_improved/checkpoints/checkpoint_10000/hyperparameters.json
```

## Training Configuration Details

The training script uses the following default configuration:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Dataset** |
| Dataset Location | `lerobot_datasets/g1_dataset/` | Converted LeRobot dataset |
| Number of Episodes | 150 | Total episodes in dataset |
| **Training** |
| Batch Size | 8 | Number of samples per batch |
| Learning Rate | 1e-4 | Initial learning rate |
| Training Steps | 100,000 | Total training iterations |
| Device | CUDA | GPU device for training |
| **Policy** |
| Policy Type | Diffusion | Diffusion Policy architecture |
| Observation Steps | 2 | Number of observation frames |
| Horizon | 16 | Number of actions to predict |
| Action Steps | 8 | Actions to execute per call |
| **Logging & Checkpoints** |
| Log Frequency | 100 steps | Print training metrics |
| Save Frequency | 10,000 steps | Save checkpoint |

## Customizing Training Parameters

To create your own configuration, edit the `CONFIGS` dictionary in `train_g1_diffusion.py`:

```python
CONFIGS = {
    "baseline": { ... },
    "improved": { ... },
    "my_config": {
        "name": "my_config",
        "batch_size": 12,
        "num_workers": 4,
        "learning_rate": 3e-5,
        "training_steps": 150_000,
        "log_freq": 100,
        "save_freq": 10_000,
        "output_dir": "outputs/train/g1_diffusion_my_config",
        "weight_decay": 3e-6,
        "gradient_clip": 7.0,
        "warmup_steps": 750,
    }
}
```

Policy configuration (same for all configs):
```python
N_OBS_STEPS = 2             # Number of observation frames
HORIZON = 16                # Action prediction horizon
N_ACTION_STEPS = 8          # Actions to execute
```

Then run with your custom config:
```bash
./run_train_g1.sh --config my_config
```

## Output Structure

Training outputs are saved to `outputs/train/g1_diffusion/`:

```
outputs/train/g1_diffusion/
├── train_g1_diffusion.log          # Training logs
└── checkpoints/
    ├── checkpoint_10000/            # Checkpoint at step 10k
    ├── checkpoint_20000/            # Checkpoint at step 20k
    ├── ...
    ├── best/                        # Best model (lowest loss)
    └── final/                       # Final trained model
```

## Monitoring Training

### View Live Logs

```bash
tail -f train_g1_diffusion.log
```

### Check Training Progress

The script prints:
- Current step and total steps
- Training loss
- Learning rate
- Checkpoint saves

Example output:
```
Step 100/100000 | Loss: 0.1234 | LR: 1.00e-04
Step 200/100000 | Loss: 0.1123 | LR: 9.99e-05
...
Saving checkpoint to outputs/train/g1_diffusion/checkpoints/checkpoint_10000
```

## Troubleshooting

### Out of Memory Error

Reduce batch size in `train_g1_diffusion.py`:
```python
BATCH_SIZE = 4  # or even 2
```

### Training Too Slow

Increase number of workers:
```python
NUM_WORKERS = 8  # Use more CPU cores for data loading
```

### Training Unstable / Loss Not Decreasing

Try adjusting learning rate:
```python
LEARNING_RATE = 5e-5  # Lower learning rate
```

## Loading a Trained Policy

After training, load your policy for evaluation:

```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load from checkpoint
policy = DiffusionPolicy.from_pretrained(
    "outputs/train/g1_diffusion/checkpoints/best"
)
```

## Next Steps

1. **Resume Training**: To continue from a checkpoint, modify the script to load the training state
2. **Evaluate Policy**: Test your trained policy on the robot or in simulation
3. **Fine-tune**: Adjust hyperparameters based on initial results

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, etc.)
- **CPU**: Multi-core processor (for data loading)
- **RAM**: 32GB+ recommended
- **Storage**: ~5GB for model checkpoints

## Expected Training Time

- **Per 1,000 steps**: ~5-10 minutes (depending on GPU)
- **Full training (100k steps)**: ~8-16 hours
- **Checkpoints saved every**: 10,000 steps (~1 hour)

## Questions?

Check the LeRobot documentation: https://github.com/huggingface/lerobot

