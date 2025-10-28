# Quick Start: Real Robot Deployment

## What Was Created

Three new files for real robot deployment:

1. **`eval_real_g1.py`** - Main deployment script
2. **`DEPLOYMENT_GUIDE.md`** - Comprehensive deployment guide
3. **`test_deployment_setup.py`** - Setup verification script

## Quick Start (3 Steps)

### Step 1: Test Your Setup

```bash
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot

# Verify everything is ready
python test_deployment_setup.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best
```

This will check:
- ✓ Checkpoint can be loaded
- ✓ Robot SDK is available  
- ✓ Image client works
- ✓ Policy inference works

### Step 2: Start Image Server (On Robot)

```bash
# SSH into the robot
ssh unitree@<ROBOT_IP>

# Start image server (captures and streams camera feeds)
cd ~/zenavatar/unitree_IL_lerobot
python unitree_lerobot/eval_robot/image_server/image_server.py
```

**Keep this running in a separate terminal!**

### Step 3: Deploy Your Policy

```bash
# On your development machine
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot

# Start deployment
python eval_real_g1.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best

# The script will:
# 1. Load the policy
# 2. Connect to robot and cameras
# 3. Show safety warnings
# 4. Wait for your confirmation
# 5. Start the control loop
```

**Press Ctrl+C to stop at any time!**

## Example Commands

### Conservative Test (Recommended First)
```bash
# 10 Hz, 10 seconds, with smoothing
python eval_real_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --frequency 10 \
    --duration 10
```

### Standard Deployment
```bash
# 10 Hz, infinite duration
python eval_real_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --frequency 10
```

### Higher Frequency (After Testing)
```bash
# 20 Hz for more responsive control
python eval_real_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --frequency 20
```

## Safety Checklist

Before running on real robot:

- [ ] Image server is running on robot
- [ ] Robot is in safe starting configuration
- [ ] Workspace is clear of obstacles
- [ ] Emergency stop is accessible
- [ ] You're ready to press Ctrl+C
- [ ] Test script passed all checks

## Getting Help

- **Setup issues**: Run `test_deployment_setup.py` for diagnostics
- **Detailed guide**: Read `DEPLOYMENT_GUIDE.md`
- **Troubleshooting**: See "Troubleshooting" section in `DEPLOYMENT_GUIDE.md`

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--frequency` | 10 | Control loop frequency (Hz) |
| `--duration` | None | Max deployment time (seconds) |
| `--motion-mode` | False | Use motion mode control |
| `--simulation` | False | Run in simulation |
| `--no-smoothing` | False | Disable action smoothing |

## What Happens During Deployment

1. **Initialization** (2-3 seconds)
   - Load policy checkpoint
   - Connect to robot
   - Connect to camera feed

2. **Safety Check**
   - Display warnings
   - Wait for user confirmation

3. **Control Loop**
   - Capture camera images
   - Get joint positions
   - Run policy inference
   - Smooth actions
   - Send commands to robot
   - Repeat at specified frequency

4. **Logging** (every 10 steps)
   - Step count
   - State range
   - Action range

## Emergency Stop

**Press Ctrl+C at any time to stop the deployment safely.**

The script will:
1. Stop sending commands
2. Clean up resources
3. Exit gracefully

## Next Steps

1. ✓ Run `test_deployment_setup.py`
2. ✓ Start image server on robot
3. ✓ Run short test deployment (10 seconds)
4. ✓ If successful, run longer deployment
5. ✓ Adjust parameters as needed

---

**Remember: Start conservative, test incrementally, safety first!**

