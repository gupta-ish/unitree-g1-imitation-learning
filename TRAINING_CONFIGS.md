# Training Configurations for G1 Diffusion Policy

This document describes the available training configurations and their differences.

## Available Configurations

### 1. Baseline Configuration
**Usage:** `bash run_train_g1.sh` or `python train_g1_diffusion.py --config baseline`

**Hyperparameters:**
- Training Steps: 100,000
- Learning Rate: 1e-4
- Batch Size: 8
- Weight Decay: 1e-6
- Gradient Clip: 10.0
- Warmup Steps: 500
- Output Directory: `outputs/train/g1_diffusion_baseline`
- Log File: `train_g1_diffusion_baseline.log`

**Use Case:** Quick baseline training for testing and validation

### 2. Improved Configuration (Recommended)
**Usage:** `bash run_train_g1_improved.sh` or `python train_g1_diffusion.py --config improved`

**Hyperparameters:**
- Training Steps: 250,000 (2.5x longer)
- Learning Rate: 5e-5 (2x lower for finer optimization)
- Batch Size: 16 (2x larger for more stable gradients)
- Weight Decay: 5e-6 (higher for better generalization)
- Gradient Clip: 5.0 (tighter for stability)
- Warmup Steps: 1,000 (longer for stability)
- Output Directory: `outputs/train/g1_diffusion_improved`
- Log File: `train_g1_diffusion_improved.log`

**Use Case:** Extended training for best performance and generalization

## Key Improvements in "Improved" Configuration

1. **Lower Learning Rate (5e-5 vs 1e-4)**
   - Allows for finer-grained optimization
   - Reduces risk of overshooting optimal parameters
   - Better convergence in later training stages

2. **Larger Batch Size (16 vs 8)**
   - More stable gradient estimates
   - Better normalization statistics
   - Smoother training dynamics

3. **More Training Steps (250k vs 100k)**
   - More time to learn complex behaviors
   - Better generalization to unseen scenarios
   - Allows the lower learning rate to fully optimize

4. **Higher Weight Decay (5e-6 vs 1e-6)**
   - Better regularization
   - Prevents overfitting to training data
   - Improves generalization to deployment

5. **Tighter Gradient Clipping (5.0 vs 10.0)**
   - More conservative updates
   - Better training stability
   - Prevents gradient explosions

6. **Longer Warmup (1000 vs 500 steps)**
   - Gradual learning rate increase
   - Prevents early training instability
   - Better initialization of batch norm statistics

## Training Speed Comparison

- **Baseline**: ~18 iterations/second on RTX 4090
- **Improved**: ~12-15 iterations/second on RTX 4090 (slower due to larger batch)

## Checkpoint Locations

Checkpoints are saved every 10,000 steps to:
- Baseline: `outputs/train/g1_diffusion_baseline/checkpoints/`
- Improved: `outputs/train/g1_diffusion_improved/checkpoints/`

Special checkpoints:
- `checkpoint_<step>`: Regular checkpoint at that step
- `best`: Best model based on training loss
- `final`: Final model after training completes

## Monitoring Training

View logs in real-time:
```bash
# Baseline
tail -f train_g1_diffusion_baseline.log

# Improved
tail -f train_g1_diffusion_improved.log
```

## Recommendations

1. **For Development/Testing**: Use baseline configuration for quick iterations
2. **For Deployment**: Use improved configuration for best performance
3. **For Experiments**: Create new configurations in `CONFIGS` dict in `train_g1_diffusion.py`

## Memory Requirements

- Baseline: ~10-12 GB GPU memory
- Improved: ~16-18 GB GPU memory (due to larger batch size)

Your RTX 4090 (24 GB) can handle both configurations comfortably.

