# Real Robot Deployment Guide

## Overview

This guide explains how to deploy your trained Diffusion Policy checkpoint on the real Unitree G1 robot.

## Prerequisites

### 1. Hardware Setup
- Unitree G1 robot powered on and in operational mode
- Robot onboard computer connected to network
- Your development machine connected to the same network as the robot
- Emergency stop accessible

### 2. Software Requirements
```bash
# All dependencies from the training environment
pip install -e unitree_lerobot/lerobot
pip install -e .

# Additional robot SDK (should already be installed)
# unitree_sdk2py
```

### 3. Network Setup
Ensure your development machine can communicate with the robot's onboard computer:
```bash
# Test connection (replace with robot's IP)
ping <ROBOT_IP>
```

## Deployment Steps

### Step 1: Start Image Server on Robot

SSH into the robot's onboard computer and start the image server:

```bash
# SSH into robot
ssh unitree@<ROBOT_IP>

# Navigate to the project directory
cd ~/zenavatar/unitree_IL_lerobot

# Start image server (runs camera capture and streaming)
python unitree_lerobot/eval_robot/image_server/image_server.py
```

**Note**: The image server must be running before starting the deployment script. It captures camera feeds and sends them to your development machine.

### Step 2: Verify Robot State

Before deployment:
1. **Visual inspection**: Check for obstacles in workspace
2. **Joint positions**: Ensure robot is in a safe starting configuration
3. **Emergency stop**: Verify emergency stop is accessible and functional
4. **Workspace**: Clear the area around the robot

### Step 3: Run Deployment Script

On your development machine:

```bash
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot

# Basic deployment (10 Hz, infinite duration)
python eval_real_g1.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best

# Deployment with custom frequency (20 Hz)
python eval_real_g1.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best --frequency 20

# Deployment with time limit (30 seconds)
python eval_real_g1.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best --duration 30

# Deployment without action smoothing (not recommended for first run)
python eval_real_g1.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best --no-smoothing
```

### Step 4: Monitor Deployment

The script will:
1. Load the policy checkpoint
2. Initialize robot interface and camera client
3. Display safety warnings
4. Wait for your confirmation ('s' to start, 'q' to quit)
5. Begin the control loop

**During deployment**:
- Monitor the console output for any warnings
- Watch the robot's movements
- Be ready to press **Ctrl+C** for emergency stop
- The script logs every 10 steps with state and action ranges

## Safety Features

### Built-in Safety Mechanisms

1. **Action Smoothing**: Exponential smoothing prevents sudden movements
   - Smoothing factor: 0.7 (configurable in code)
   
2. **Action Delta Limiting**: Maximum change per step is clamped
   - Max delta: 0.1 radians per step (configurable in code)
   
3. **User Confirmation**: Script requires explicit confirmation before starting

4. **Emergency Stop**: Press Ctrl+C at any time to stop

### Customizing Safety Parameters

Edit `eval_real_g1.py` to adjust:
```python
# In RealRobotDeployment.__init__()
self.max_action_delta = 0.1  # Maximum change per step (radians)
self.action_smoothing_alpha = 0.7  # Smoothing factor (0=no smoothing, 1=no history)
```

## Command-Line Arguments

```bash
python eval_real_g1.py --help
```

### Required Arguments
- `--checkpoint`: Path to checkpoint directory

### Optional Arguments
- `--frequency`: Control loop frequency in Hz (default: 10)
- `--duration`: Maximum deployment duration in seconds (default: infinite)
- `--arm-type`: Robot arm type - G1_29 or G1_23 (default: G1_29)
- `--motion-mode`: Use motion mode for robot control (default: False)
- `--simulation`: Run in simulation mode for testing (default: False)
- `--no-smoothing`: Disable action smoothing (not recommended)

## Troubleshooting

### Problem: "Cannot connect to image server"

**Solution**:
1. Verify image server is running on robot
2. Check network connectivity: `ping <ROBOT_IP>`
3. Ensure firewall allows the connection

### Problem: "Robot not responding"

**Solution**:
1. Check robot power and initialization
2. Verify SDK is properly installed
3. Check motion mode setting (try with/without `--motion-mode`)

### Problem: "Policy predictions look wrong"

**Solution**:
1. Verify checkpoint path is correct
2. Check camera calibration matches training data
3. Ensure camera feed is from correct camera (cam_left_high)
4. Verify robot starting pose matches training data starting pose

### Problem: "Robot movements are jerky"

**Solution**:
1. Enable action smoothing (remove `--no-smoothing` if used)
2. Decrease control frequency: `--frequency 5`
3. Increase smoothing factor in code
4. Check network latency between machines

### Problem: "Actions are too slow/conservative"

**Solution**:
1. Try disabling action smoothing: `--no-smoothing`
2. Increase max_action_delta in code
3. Increase control frequency: `--frequency 20`

## Testing in Simulation

Before deploying on real hardware, test in simulation:

```bash
# Run in simulation mode
python eval_real_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --simulation \
    --duration 30
```

## Best Practices

1. **Start Conservative**: Begin with low frequency (10 Hz) and action smoothing enabled
2. **Short Duration First**: Test with `--duration 10` for 10-second runs initially
3. **Gradual Increase**: Gradually increase frequency and duration as you gain confidence
4. **Multiple Checkpoints**: Test different checkpoints (best, final, specific steps)
5. **Document Results**: Keep notes on what works and what doesn't
6. **Backup Plan**: Always have emergency stop ready

## Emergency Procedures

### If Robot Makes Unexpected Movement
1. **Immediately press Ctrl+C** to stop the script
2. Activate physical emergency stop if needed
3. Check logs for error messages
4. Review the last few actions in console output

### If Script Crashes
1. The robot will stop receiving commands (safe state)
2. Restart image server if needed
3. Check error logs
4. Restart deployment script

## Performance Optimization

### Reducing Latency
1. **Network**: Use wired connection instead of WiFi
2. **Frequency**: Balance between responsiveness and stability
3. **Image Size**: Camera resolution matches training data (480x640)

### Improving Policy Performance
1. **Initialization**: Ensure robot starts in same pose as training episodes
2. **Environment**: Match lighting and object positions to training data
3. **Calibration**: Verify camera extrinsics match training setup

## Example Workflow

```bash
# Terminal 1: On robot
ssh unitree@<ROBOT_IP>
cd ~/zenavatar/unitree_IL_lerobot
python unitree_lerobot/eval_robot/image_server/image_server.py

# Terminal 2: On development machine
cd /home/ishitagupta/zenavatar/unitree_IL_lerobot

# Short test run
python eval_real_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --frequency 10 \
    --duration 10

# If successful, run longer
python eval_real_g1.py \
    --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best \
    --frequency 10 \
    --duration 60
```

## Contact and Support

- Review robot logs in the unitree SDK
- Check camera feed quality
- Verify policy checkpoint integrity
- Test with different episodes' starting poses

## Success Metrics

Monitor these during deployment:
- **Smooth movements**: No jerky or erratic behavior
- **Task completion**: Robot completes intended task
- **Safety**: No collisions or dangerous movements
- **Stability**: Consistent performance over time
- **Responsiveness**: Actions match expected behavior

---

**Remember**: Safety first! Always be prepared to emergency stop the robot.

