# ACT (Action Chunking Transformers) Training Guide

## Overview

Action Chunking Transformers (ACT) is a powerful policy architecture that predicts chunks of actions using transformer encoders/decoders with an optional variational objective. ACT has shown excellent performance on bimanual manipulation tasks.

**Paper:** Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware  
**Link:** https://huggingface.co/papers/2304.13705

## Key Features of ACT

### 1. **Action Chunking**
- Predicts sequences of actions (chunks) instead of single actions
- Default: 100 action steps per prediction
- Reduces compounding errors and improves temporal consistency

### 2. **Transformer Architecture**
- Vision backbone: ResNet18 (pretrained on ImageNet)
- Transformer encoder: Processes observations
- Transformer decoder: Generates action sequences
- Cross-attention: Conditions actions on observations

### 3. **Variational Objective (VAE)**
- Optional VAE encoder for learning latent representations
- Helps with multimodal action distributions
- KL-divergence regularization

### 4. **Temporal Ensembling** (Optional)
- Exponentially weighted ensemble of predictions
- Can be enabled for inference (not in training)

## Training Configurations

### Baseline Configuration
```bash
python train_g1_act.py --config baseline
# or
./run_train_g1_act.sh baseline
```

**Parameters:**
- Training steps: 100,000
- Batch size: 8
- Learning rate: 1e-5 (1e-5 for backbone)
- Chunk size: 100
- Model dimension: 512
- Attention heads: 8
- Encoder layers: 4
- Decoder layers: 1
- Latent dimension: 32
- VAE: Enabled
- KL weight: 10.0

**Estimated training time:** ~8-12 hours on RTX 3090

**Output:** `outputs/train/g1_act_baseline/`

### Large Configuration
```bash
python train_g1_act.py --config large
# or
./run_train_g1_act.sh large
```

**Parameters:**
- Training steps: 200,000
- Batch size: 8
- Learning rate: 5e-6 (1e-6 for backbone)
- Chunk size: 100
- Model dimension: 1024 (2x larger)
- Attention heads: 16 (2x more)
- Encoder layers: 6 (more capacity)
- Decoder layers: 1
- Latent dimension: 64 (2x larger)
- VAE: Enabled
- KL weight: 10.0

**Estimated training time:** ~20-30 hours on RTX 3090

**Output:** `outputs/train/g1_act_large/`

## Quick Start

### Step 1: Verify Environment

```bash
# Activate environment
conda activate unitree_IL

# Check dataset
ls -lh /home/ishitagupta/zenavatar/lerobot_datasets/your_name/g1_dataset/

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Start Training

```bash
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot

# Start baseline training
./run_train_g1_act.sh baseline

# OR start large model training
./run_train_g1_act.sh large
```

### Step 3: Monitor Training

```bash
# Watch log file
tail -f train_g1_act_baseline.log

# OR for large config
tail -f train_g1_act_large.log
```

## Training Outputs

### Checkpoints
Saved every 10,000 steps:
```
outputs/train/g1_act_baseline/checkpoints/
├── checkpoint_10000/
├── checkpoint_20000/
├── ...
├── checkpoint_100000/
├── best/              # Best model (lowest loss)
└── final/             # Final model
```

### Each Checkpoint Contains:
- `config.json` - Policy configuration
- `model.safetensors` - Model weights (~400MB for baseline, ~1.5GB for large)
- `training_state.pt` - Optimizer and scheduler state
- `hyperparameters.json` - Training hyperparameters

### Log File
- `train_g1_act_baseline.log` - Full training log
- Real-time loss, L1 loss, KLD loss, learning rate
- Logged every 100 steps

## Understanding the Logs

### Example Log Output:
```
Step 10000/100000 | Loss: 0.0234 | L1: 0.0198 | KLD: 0.0036 | LR: 1.00e-05
```

**Metrics:**
- **Loss**: Total loss (L1 + KL_weight × KLD)
- **L1**: L1 reconstruction loss between predicted and ground truth actions
- **KLD**: KL-divergence loss (only when VAE is enabled)
- **LR**: Current learning rate

### Good Training Signs:
- ✓ Steady decrease in Loss and L1
- ✓ KLD stabilizes around 0.001-0.01
- ✓ No sudden spikes or NaN values
- ✓ L1 loss < 0.05 indicates good action prediction

### Warning Signs:
- ⚠️ Loss increases or plateaus early
- ⚠️ KLD becomes negative or very large
- ⚠️ NaN or Inf in losses
- ⚠️ GPU out of memory errors

## ACT vs Diffusion Policy

| Feature | ACT | Diffusion |
|---------|-----|-----------|
| **Architecture** | Transformer | U-Net + Diffusion Process |
| **Prediction** | Direct action chunks | Iterative denoising |
| **Speed** | Fast inference (single forward pass) | Slower (multiple denoising steps) |
| **Training** | Faster convergence | May need more steps |
| **Memory** | Moderate | Higher (needs noise schedule) |
| **Multimodality** | VAE for multimodal actions | Natural multimodality via diffusion |
| **Best for** | Precise, deterministic tasks | Complex, multimodal behaviors |

## Hyperparameter Tuning

### If training is unstable:
1. **Reduce learning rate:**
   ```python
   "learning_rate": 5e-6  # Try 5e-6 or 1e-6
   ```

2. **Increase warmup steps:**
   ```python
   "warmup_steps": 2000  # More gradual warmup
   ```

3. **Adjust KL weight:**
   ```python
   "kl_weight": 5.0  # Reduce if KLD dominates
   ```

### If actions are not accurate:
1. **Increase model capacity:**
   ```python
   "dim_model": 1024
   "n_encoder_layers": 6
   ```

2. **Train longer:**
   ```python
   "training_steps": 200_000
   ```

3. **Adjust chunk size:**
   ```python
   "chunk_size": 50  # Shorter chunks for faster feedback
   ```

## Evaluation

After training, evaluate your ACT policy:

```bash
# Evaluate on dataset
python evaluate_g1_policy.py \
    --checkpoint outputs/train/g1_act_baseline/checkpoints/best \
    --episode 5 \
    --create-video
```

## Deployment on Real Robot

After training, you can deploy ACT on the real robot using the same deployment script:

```bash
python eval_real_g1.py \
    --checkpoint outputs/train/g1_act_baseline/checkpoints/best \
    --frequency 10
```

**Note:** ACT predicts action chunks, so the robot will execute actions smoothly over the chunk horizon.

## Troubleshooting

### Problem: GPU Out of Memory

**Solutions:**
1. Reduce batch size:
   ```python
   "batch_size": 4  # or 2
   ```

2. Use smaller model:
   ```bash
   ./run_train_g1_act.sh baseline  # Instead of large
   ```

3. Reduce model dimensions:
   ```python
   "dim_model": 256  # Smaller hidden dimension
   ```

### Problem: Loss not decreasing

**Solutions:**
1. Check learning rate (may be too high or too low)
2. Verify dataset is loading correctly
3. Increase warmup steps
4. Check if KL weight is too high (overwhelming reconstruction loss)

### Problem: KLD loss is negative or very large

**Solutions:**
1. Adjust KL weight:
   ```python
   "kl_weight": 5.0  # or 20.0
   ```

2. Check VAE latent dimension:
   ```python
   "latent_dim": 32  # Try different values
   ```

3. Consider disabling VAE:
   ```python
   "use_vae": False
   ```

### Problem: Training is too slow

**Solutions:**
1. Increase num_workers for data loading:
   ```python
   "num_workers": 8  # or more
   ```

2. Use smaller image resolution (requires dataset changes)

3. Train on multiple GPUs (requires code modification)

## Advanced Configuration

### Custom Configuration

Create your own configuration in `train_g1_act.py`:

```python
CONFIGS["custom"] = {
    "name": "custom",
    "batch_size": 16,
    "learning_rate": 3e-5,
    "training_steps": 150_000,
    # ... other parameters
    "chunk_size": 50,
    "dim_model": 768,
    "n_encoder_layers": 5,
}
```

Then run:
```bash
python train_g1_act.py --config custom
```

## Performance Benchmarks

### Expected Results (based on similar tasks):

| Metric | Baseline (100k steps) | Large (200k steps) |
|--------|----------------------|-------------------|
| Final Loss | ~0.020 - 0.030 | ~0.015 - 0.025 |
| L1 Loss | ~0.015 - 0.025 | ~0.012 - 0.020 |
| Training Time | 8-12 hours | 20-30 hours |
| Model Size | ~400 MB | ~1.5 GB |
| Inference Speed | ~100 Hz | ~50 Hz |

**Note:** Actual results depend on dataset quality, task complexity, and hardware.

## Best Practices

1. ✓ **Start with baseline config** - Validate setup before trying larger models
2. ✓ **Monitor losses closely** - Both L1 and KLD should decrease
3. ✓ **Save checkpoints frequently** - Can resume if training is interrupted
4. ✓ **Evaluate on held-out episodes** - Check generalization
5. ✓ **Compare with Diffusion** - Try both policies and pick the best
6. ✓ **Use pretrained backbone** - ImageNet weights help with vision
7. ✓ **Adjust chunk size for your task** - Longer chunks for smooth tasks

## Citation

If you use ACT in your research:

```bibtex
@inproceedings{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```

## Next Steps

1. ✅ Start training with baseline config
2. ✅ Monitor training progress
3. ✅ Evaluate trained model on dataset
4. ✅ Compare results with Diffusion policy
5. ✅ Deploy best model on real robot
6. ✅ Fine-tune hyperparameters if needed

---

**Happy Training! 🚀**

For questions or issues, check the logs and refer to the troubleshooting section above.

