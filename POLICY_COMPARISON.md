# Policy Comparison: ACT vs Diffusion

## Quick Comparison

| Aspect | ACT | Diffusion |
|--------|-----|-----------|
| **Training Script** | `train_g1_act.py` | `train_g1_diffusion.py` |
| **Launch Script** | `./run_train_g1_act.sh` | `./run_train_g1.sh` |
| **Architecture** | Transformer Encoder-Decoder | U-Net with Diffusion Process |
| **Action Prediction** | Single forward pass → action chunk | Iterative denoising → action chunk |
| **Inference Speed** | ⚡ Fast (~100 Hz) | Moderate (~30 Hz) |
| **Training Speed** | ⚡ Faster convergence | Slower, needs more steps |
| **Memory Usage** | Moderate | Higher |
| **Multimodality** | VAE-based | Natural via diffusion |
| **Best For** | Precise, fast execution | Complex, multimodal behaviors |

## Architecture Details

### ACT (Action Chunking Transformers)
```
Input (Images + State)
    ↓
ResNet18 (Vision Backbone)
    ↓
Transformer Encoder (4-6 layers)
    ↓ (cross-attention)
Transformer Decoder (1 layer)
    ↓
Action Chunk (100 steps)

Optional: VAE encoder for latent sampling
```

**Key Components:**
- Vision: ResNet18 pretrained on ImageNet
- Attention: Multi-head self-attention and cross-attention
- Output: Direct prediction of action sequences
- VAE: Optional variational objective for multimodality

### Diffusion Policy
```
Input (Images + State)
    ↓
ResNet18 (Vision Backbone)
    ↓
U-Net (with noise level conditioning)
    ↓ (iterative denoising)
Denoising Process (10-100 steps)
    ↓
Action Chunk (16 steps)
```

**Key Components:**
- Vision: ResNet18 pretrained on ImageNet
- Noise: DDPM/DDIM scheduler
- Output: Iterative denoising from Gaussian noise
- Process: Multiple forward passes for refinement

## Training Configurations

### ACT Baseline
```yaml
Training Steps: 100,000 (faster to converge)
Batch Size: 8
Learning Rate: 1e-5
Chunk Size: 100 actions
Model Dim: 512
Training Time: ~8-12 hours
Model Size: ~400 MB
```

### Diffusion Baseline
```yaml
Training Steps: 100,000
Batch Size: 8
Learning Rate: 1e-4 (10x higher)
Horizon: 16 actions (smaller chunks)
Model Dim: 256
Training Time: ~10-15 hours
Model Size: ~200 MB
```

### ACT Large
```yaml
Training Steps: 200,000
Model Dim: 1024 (2x larger)
Attention Heads: 16
Encoder Layers: 6
Training Time: ~20-30 hours
Model Size: ~1.5 GB
```

### Diffusion Improved
```yaml
Training Steps: 250,000 (2.5x longer)
Batch Size: 16
Learning Rate: 5e-5
Training Time: ~25-35 hours
Model Size: ~250 MB
```

## Performance Characteristics

### ACT Strengths ✓
- **Fast inference**: Single forward pass
- **Efficient training**: Faster convergence
- **Long horizon**: Predicts 100 action steps
- **Temporal consistency**: Smoother action sequences
- **Proven on bimanual tasks**: Designed for dual-arm manipulation

### ACT Weaknesses ✗
- **Limited multimodality**: VAE may not capture all modes
- **Fixed chunks**: Less flexible than diffusion
- **Larger models**: More parameters for similar performance

### Diffusion Strengths ✓
- **Natural multimodality**: Diffusion process handles multiple solutions
- **Flexible**: Can adjust denoising steps at inference
- **Robust**: Less sensitive to hyperparameters
- **State-of-the-art**: Recent advances in diffusion models

### Diffusion Weaknesses ✗
- **Slower inference**: Multiple denoising steps required
- **Training time**: Needs more training steps
- **Memory**: Higher memory usage during training
- **Tuning**: Requires careful noise schedule tuning

## Which to Choose?

### Choose ACT if:
- ✓ You need **fast, real-time control** (>50 Hz)
- ✓ Your task is **deterministic** (single solution)
- ✓ You want **faster training** and iteration
- ✓ You have **bimanual manipulation** tasks
- ✓ You need **long action sequences** (100+ steps)

### Choose Diffusion if:
- ✓ Your task has **multiple valid solutions** (multimodal)
- ✓ **Inference speed** is not critical (<30 Hz OK)
- ✓ You want **state-of-the-art** performance
- ✓ You can afford **longer training time**
- ✓ You want **more flexibility** at inference time

### Our Recommendation for Unitree G1:
1. **Start with ACT baseline** - Fast to train, good for bimanual tasks
2. **Train Diffusion improved** - Longer training but potentially better
3. **Compare results** - Evaluate both on your specific task
4. **Pick the winner** - Deploy the better-performing model

## Training Commands

### Train Both Policies

```bash
# Terminal 1: Train ACT (baseline)
./run_train_g1_act.sh baseline

# Terminal 2: Train Diffusion (improved)
./run_train_g1.sh improved
```

### Or train sequentially:

```bash
# Train ACT first (8-12 hours)
./run_train_g1_act.sh baseline

# Then train Diffusion (25-35 hours)
./run_train_g1.sh improved
```

## Evaluation Commands

```bash
# Evaluate ACT
python evaluate_g1_policy.py \
    --checkpoint outputs/train/g1_act_baseline/checkpoints/best \
    --episode 5 \
    --create-video

# Evaluate Diffusion
python evaluate_g1_policy.py \
    --checkpoint outputs/train/g1_diffusion_improved/checkpoints/best \
    --episode 5 \
    --create-video
```

## Real Robot Deployment

Both policies use the same deployment script:

```bash
# Deploy ACT
python eval_real_g1.py \
    --checkpoint outputs/train/g1_act_baseline/checkpoints/best \
    --frequency 10

# Deploy Diffusion
python eval_real_g1.py \
    --checkpoint outputs/train/g1_diffusion_improved/checkpoints/best \
    --frequency 10
```

## Expected Results

Based on similar manipulation tasks:

### ACT
- **L1 Loss**: 0.015 - 0.025 (baseline)
- **Success Rate**: 70-85% (depends on task)
- **Inference**: ~100 Hz
- **Smoothness**: ⭐⭐⭐⭐⭐ (excellent)

### Diffusion
- **MSE Loss**: 0.002 - 0.010 (improved)
- **Success Rate**: 75-90% (depends on task)
- **Inference**: ~30 Hz
- **Smoothness**: ⭐⭐⭐⭐ (very good)

## Resource Requirements

### GPU Memory

| Config | Training | Inference |
|--------|----------|-----------|
| ACT Baseline | ~8 GB | ~2 GB |
| ACT Large | ~16 GB | ~4 GB |
| Diffusion Baseline | ~10 GB | ~3 GB |
| Diffusion Improved | ~14 GB | ~4 GB |

### Disk Space

| Config | Checkpoints (all) | Best Model |
|--------|------------------|------------|
| ACT Baseline | ~4 GB | ~400 MB |
| ACT Large | ~15 GB | ~1.5 GB |
| Diffusion Baseline | ~2 GB | ~200 MB |
| Diffusion Improved | ~6 GB | ~250 MB |

## Summary

| Metric | ACT Winner? | Diffusion Winner? |
|--------|------------|-------------------|
| Speed | ✅ | |
| Training Time | ✅ | |
| Multimodality | | ✅ |
| Flexibility | | ✅ |
| Bimanual Tasks | ✅ | |
| State-of-art | | ✅ |
| Simplicity | ✅ | |

**Bottom Line:** Both are excellent choices! Train both and let the results decide. 🚀

---

## Quick Start Both

```bash
# 1. Start ACT training
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot
./run_train_g1_act.sh baseline

# 2. Monitor progress
tail -f train_g1_act_baseline.log

# 3. After ACT finishes, compare with your existing Diffusion model
# You already have: outputs/train/g1_diffusion_improved/checkpoints/best

# 4. Evaluate both
python evaluate_g1_policy.py --checkpoint <ACT_CHECKPOINT>
python evaluate_g1_policy.py --checkpoint <DIFFUSION_CHECKPOINT>

# 5. Deploy the better one!
```

Happy training! 🎯

