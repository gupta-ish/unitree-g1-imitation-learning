# Training Configurations

Simple guide to training with different configurations.

## Quick Start

**Baseline (faster, for testing):**
```bash
./run_train_g1.sh --config baseline
```

**Improved (longer, better performance):**
```bash
./run_train_g1.sh --config improved
```

## What's Different?

| Feature | Baseline | Improved |
|---------|----------|----------|
| Steps | 100,000 | 250,000 |
| Learning Rate | 1e-4 | 5e-5 |
| Batch Size | 8 | 16 |
| Time (RTX 4090) | ~5-6 hours | ~17-20 hours |
| Use Case | Quick testing | Best performance |

## Checkpoints Include Hyperparameters

Every checkpoint automatically saves its hyperparameters:

```bash
# View what settings were used
cat outputs/train/g1_diffusion_improved/checkpoints/checkpoint_10000/hyperparameters.json
```

Example `hyperparameters.json`:
```json
{
  "training_config": {
    "learning_rate": 5e-05,
    "batch_size": 16,
    "training_steps": 250000,
    ...
  },
  "policy_config": {
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8
  },
  "training_step": 10000,
  "loss": 0.0234
}
```

## Add Your Own Config

Edit `train_g1_diffusion.py` and add to the `CONFIGS` dictionary:

```python
"my_config": {
    "name": "my_config",
    "batch_size": 12,
    "learning_rate": 7e-5,
    "training_steps": 150_000,
    "log_freq": 100,
    "save_freq": 10_000,
    "output_dir": "outputs/train/g1_diffusion_my_config",
    "weight_decay": 3e-6,
    "gradient_clip": 7.0,
    "warmup_steps": 750,
}
```

Then run: `./run_train_g1.sh --config my_config`

