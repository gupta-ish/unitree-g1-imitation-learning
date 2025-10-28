## Evaluating Your Trained Diffusion Policy

This guide explains how to test and visualize your trained policy **without deploying to the physical robot**.

---

## Overview

You have three main evaluation tools:

1. **`evaluate_g1_policy.py`** - Detailed single-episode evaluation with visualizations
2. **`batch_evaluate_g1.py`** - Evaluate on multiple episodes for aggregate statistics  
3. **Simulation** (if available) - Test in a physics simulator

---

## 1. Single Episode Evaluation

Evaluate the policy on one episode and generate detailed visualizations.

### Basic Usage

```bash
python evaluate_g1_policy.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --episode 5
```

### With Video Rollout

```bash
python evaluate_g1_policy.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --episode 5 \
    --create-video \
    --output-dir outputs/evaluation/episode_5
```

### What You Get

The evaluation produces:

#### 1. **Action Comparison Plots** (`action_comparison_epX.png`)
- Side-by-side comparison of predicted vs ground truth actions
- One subplot per robot joint
- Shows how well the policy reproduces the demonstrated behavior

#### 2. **Error Distribution Plots** (`error_distribution_epX.png`)
- Error over time for all joints
- Box plots showing error distribution per joint
- Helps identify which joints are harder to predict

#### 3. **Video Rollout** (`rollout_epX.mp4`) [optional]
- Shows camera observations alongside action predictions
- Real-time comparison of predicted vs ground truth
- Great for presentations and debugging

#### 4. **Evaluation Report** (`evaluation_report.txt`)
```
OVERALL METRICS
Mean Squared Error (MSE):  0.001234
Mean Absolute Error (MAE): 0.023456
Root Mean Squared Error:   0.035123

PER-JOINT METRICS
Joint Name           MSE         MAE
kLeftShoulderPitch   0.001234   0.023456
kLeftShoulderRoll    0.002345   0.034567
...
```

---

## 2. Batch Evaluation

Test on multiple episodes to get statistically meaningful results.

### Usage

```bash
# Evaluate on first 20 episodes
python batch_evaluate_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --num-episodes 20 \
    --start-episode 0 \
    --output-dir outputs/batch_evaluation
```

### Output

Creates `batch_results.json` with:

```json
{
  "aggregate_stats": {
    "num_episodes_evaluated": 20,
    "mse": {
      "mean": 0.001234,
      "std": 0.000123,
      "min": 0.000890,
      "max": 0.001678
    },
    "mae": {
      "mean": 0.023456,
      "std": 0.002345,
      "min": 0.019012,
      "max": 0.028901
    },
    "per_joint": {
      "mse_mean": [0.001, 0.002, ...],
      "mae_mean": [0.023, 0.034, ...]
    }
  },
  "per_episode_metrics": [...]
}
```

---

## 3. Understanding the Metrics

### Mean Squared Error (MSE)
- **What**: Average of squared differences between predicted and ground truth actions
- **Good value**: < 0.01 (depends on your action scale)
- **Interpretation**: Lower is better; penalizes large errors heavily

### Mean Absolute Error (MAE)  
- **What**: Average absolute difference
- **Good value**: < 0.05 (depends on your action scale)
- **Interpretation**: More intuitive than MSE; actual average error magnitude

### Per-Joint Analysis
- Shows which joints are easier/harder to predict
- Arm joints usually have lower error than hand joints
- High error on specific joints may indicate:
  - Need more training data for those movements
  - That joint has more complex dynamics
  - Possible data quality issues

---

## 4. Comparing Checkpoints

Evaluate different checkpoints to find the best one:

```bash
# Evaluate checkpoint at step 50k
python batch_evaluate_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/checkpoint_50000 \
    --num-episodes 10 \
    --output-dir outputs/eval_50k

# Evaluate best checkpoint
python batch_evaluate_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --num-episodes 10 \
    --output-dir outputs/eval_best

# Compare results
diff outputs/eval_50k/batch_results.json outputs/eval_best/batch_results.json
```

---

## 5. Visual Inspection Checklist

When reviewing the generated plots and videos:

### ✅ Good Signs
- [ ] Predicted actions closely follow ground truth
- [ ] Smooth predicted trajectories (no jitter)
- [ ] Low error consistently across time
- [ ] Similar error across all joints
- [ ] Policy captures key movement patterns

### ⚠️ Warning Signs  
- [ ] Large spikes in error at specific timeframes
- [ ] Predicted actions lag behind ground truth
- [ ] High variance in predictions (shaky behavior)
- [ ] Some joints have much higher error than others
- [ ] Policy predictions are too smooth (underfitting)

---

## 6. Testing on Validation Episodes

To properly evaluate generalization, test on episodes NOT seen during training:

```bash
# If you have 150 episodes and trained on first 140
# Test on episodes 140-149
python batch_evaluate_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --num-episodes 10 \
    --start-episode 140 \
    --output-dir outputs/validation_eval
```

**Note**: If you trained on all 150 episodes, the evaluation shows "training performance" not "generalization". Consider:
1. Re-converting your dataset with a train/val split
2. Recording more episodes
3. Using cross-validation

---

## 7. Simulation Testing (Advanced)

If you have access to a Unitree G1 simulator (like Isaac Sim or MuJoCo):

### Setup
1. Install simulation environment
2. Create interface to send policy actions to simulator
3. Record simulation episodes for comparison

### Benefits
- Test policy in real-time without robot
- Visualize 3D robot movements
- Test edge cases and failure modes safely
- Evaluate contact physics and dynamics

### Example Pseudo-code
```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load policy
policy = DiffusionPolicy.from_pretrained("outputs/.../best")
policy.eval()

# Connect to simulator
sim = UnitreeG1Simulator()
sim.reset()

# Run policy in sim
for step in range(1000):
    obs = sim.get_observation()  # Get camera images and joint states
    action = policy.select_action(obs)
    sim.step(action)
    sim.render()
```

---

## 8. Debugging Poor Performance

If evaluation metrics are poor:

### A. Check Training
```bash
# Look at training logs
tail -n 100 train_g1_diffusion_baseline.log

# Check if loss decreased
grep "Loss:" train_g1_diffusion_baseline.log
```

### B. Visualize Predictions
```bash
# Create videos for multiple episodes
for i in {0..5}; do
    python evaluate_g1_policy.py \
        --checkpoint outputs/.../best \
        --episode $i \
        --create-video \
        --output-dir outputs/debug/ep$i
done
```

### C. Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| High error on all joints | Undertraining | Train longer or adjust hyperparameters |
| Error increases over time | Compounding errors | Increase n_obs_steps or use receding horizon |
| Some joints much worse | Imbalanced data | Collect more diverse demonstrations |
| Predictions too smooth | Over-regularization | Reduce weight_decay or lower diffusion steps |
| Jittery predictions | Insufficient training | Train longer or increase batch size |

---

## 9. Quick Evaluation Workflow

### After Training Completes

```bash
# 1. Evaluate best checkpoint on a single episode with video
python evaluate_g1_policy.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --episode 10 \
    --create-video \
    --output-dir outputs/quick_eval

# 2. Review the plots
eog outputs/quick_eval/*.png

# 3. Watch the video
vlc outputs/quick_eval/rollout_ep10.mp4

# 4. If looks good, run batch evaluation
python batch_evaluate_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --num-episodes 20 \
    --output-dir outputs/full_eval

# 5. Check aggregate metrics
cat outputs/full_eval/batch_results.json | jq '.aggregate_stats'
```

---

## 10. Exporting for Robot Deployment

Once evaluation looks good, prepare for robot deployment:

```bash
# The checkpoint directory contains everything needed:
outputs/train/g1_diffusion_baseline/checkpoints/best/
├── config.json              # Policy configuration
├── model.safetensors        # Model weights
├── hyperparameters.json     # Training hyperparameters
└── training_state.pt        # Full training state
```

### Loading on Robot

```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load policy (handles all config automatically)
policy = DiffusionPolicy.from_pretrained(
    "outputs/train/g1_diffusion_baseline/checkpoints/best"
)
policy.eval()
policy.to("cuda")  # Or "cpu" if running on robot computer

# Use in control loop
while True:
    obs = robot.get_observation()
    action = policy.select_action(obs)
    robot.execute_action(action)
```

---

## 11. Expected Performance Ranges

Based on typical robot learning benchmarks:

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **MSE** | < 0.001 | 0.001-0.01 | 0.01-0.05 | > 0.05 |
| **MAE** | < 0.01 | 0.01-0.05 | 0.05-0.1 | > 0.1 |
| **Visual** | Indistinguishable | Very close | Recognizable pattern | Different behavior |

**Note**: These depend on your action space normalization and task complexity.

---

## 12. Next Steps

After successful evaluation:

1. ✅ **Validation passed** → Deploy to real robot in safe environment
2. ⚠️ **Metrics acceptable** → Consider collecting more data or training longer
3. ❌ **Poor performance** → Debug training, check data quality, adjust hyperparameters

---

## Questions?

- Check training logs: `train_g1_diffusion_*.log`
- Review checkpoint hyperparameters: `checkpoints/*/hyperparameters.json`
- Visualize multiple episodes to understand failure modes
- Consider simulation testing before real robot deployment

