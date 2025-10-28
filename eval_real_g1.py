"""
Real Robot Deployment Script for Unitree G1 with Trained Diffusion Policy.

This script deploys a trained policy checkpoint on the real Unitree G1 robot.

Prerequisites:
1. Image server must be running on the robot's onboard computer:
   `python unitree_lerobot/eval_robot/image_server/image_server.py`

2. Robot must be powered on and in a safe operational state

Usage:
    python eval_real_g1.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best

Safety Features:
- Emergency stop on Ctrl+C
- User confirmation before initialization
- Action smoothing and velocity limits
- Error recovery mechanisms
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Motor indices for G1 dual arm (14 DOF)
MOTOR_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw"
]


class RealRobotDeployment:
    """Deployment manager for real robot policy execution."""
    
    def __init__(self, checkpoint_path: Path, arm_type="G1_29", motion_mode=False, simulation_mode=False):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.arm_type = arm_type
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.device = DEVICE
        
        # Safety parameters
        self.max_action_delta = 0.1  # Maximum change per step (radians)
        self.action_smoothing_alpha = 0.7  # Exponential smoothing factor
        self.previous_action = None
        
        # Load policy
        logging.info(f"Loading policy from {self.checkpoint_path}")
        try:
            # Try loading as local checkpoint
            self.policy = DiffusionPolicy.from_pretrained(
                str(self.checkpoint_path),
                local_files_only=True
            )
        except Exception as e:
            logging.error(f"Failed to load from local path: {e}")
            logging.info("Trying alternative loading method...")
            # Alternative: load config and weights separately
            import json
            from lerobot.configs.policies import DiffusionConfig
            from safetensors.torch import load_file
            
            config_path = self.checkpoint_path / "config.json"
            model_path = self.checkpoint_path / "model.safetensors"
            
            with open(config_path) as f:
                config_dict = json.load(f)
            
            config = DiffusionConfig(**config_dict)
            self.policy = DiffusionPolicy(config)
            state_dict = load_file(str(model_path))
            self.policy.load_state_dict(state_dict)
            
        self.policy.eval()
        self.policy.to(self.device)
        logging.info(f"Policy loaded successfully on {self.device}")
        
        # Robot interfaces (will be initialized later)
        self.robot_interface = None
        self.image_client = None
        
    def setup_robot_interface(self):
        """Initialize robot arm controller and IK solver."""
        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_ArmController
        from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK
        
        logging.info("Initializing robot interface...")
        
        # Initialize arm controller
        self.arm_ctrl = G1_29_ArmController(
            motion_mode=self.motion_mode,
            simulation_mode=self.simulation_mode
        )
        
        # Initialize IK solver
        self.arm_ik = G1_29_ArmIK()
        
        logging.info("Robot interface initialized")
        
    def setup_image_client(self):
        """Initialize image client to receive camera feeds from robot."""
        from multiprocessing import shared_memory
        import threading
        from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
        
        logging.info("Setting up image client...")
        
        # Camera configuration for G1_Dual_Arm (single head camera)
        img_shape = (480, 640, 3)  # cam_left_high
        
        # Create shared memory for images
        self.img_shm = shared_memory.SharedMemory(
            create=True, 
            size=np.prod(img_shape) * np.uint8().itemsize
        )
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.img_shm.buf)
        
        # Initialize image client
        self.image_client = ImageClient(
            tv_img_shape=img_shape,
            tv_img_shm_name=self.img_shm.name
        )
        
        # Start image receiving thread
        self.image_receive_thread = threading.Thread(
            target=self.image_client.receive_process,
            daemon=True
        )
        self.image_receive_thread.start()
        
        logging.info("Image client ready")
        
    def get_observation(self):
        """Capture current observation from robot."""
        # Get current image (HWC format from camera)
        current_image = self.img_array.copy()
        
        # Convert HWC to CHW format for policy
        if current_image.ndim == 3 and current_image.shape[2] == 3:
            current_image = np.transpose(current_image, (2, 0, 1))  # HWC -> CHW
        
        # Get current joint positions
        current_joint_positions = self.arm_ctrl.get_current_dual_arm_q()
        
        # Prepare observation dict in the format expected by policy
        observation = {
            "observation.images.cam_left_high": torch.from_numpy(current_image),
            "observation.state": torch.from_numpy(current_joint_positions).float()
        }
        
        return observation, current_joint_positions
    
    def smooth_action(self, action: np.ndarray) -> np.ndarray:
        """Apply smoothing and safety limits to action."""
        if self.previous_action is None:
            self.previous_action = action
            return action
        
        # Exponential smoothing
        smoothed_action = (self.action_smoothing_alpha * action + 
                          (1 - self.action_smoothing_alpha) * self.previous_action)
        
        # Clip action delta for safety
        action_delta = smoothed_action - self.previous_action
        action_delta = np.clip(action_delta, -self.max_action_delta, self.max_action_delta)
        final_action = self.previous_action + action_delta
        
        self.previous_action = final_action
        return final_action
    
    def execute_action(self, action: np.ndarray):
        """Send action to robot."""
        # Compute torques using IK
        tau = self.arm_ik.solve_tau(action)
        
        # Send command to robot
        self.arm_ctrl.ctrl_dual_arm(action, tau)
    
    def run(self, frequency: int = 10, duration: float | None = None, enable_smoothing: bool = True):
        """
        Main deployment loop.
        
        Args:
            frequency: Control loop frequency in Hz
            duration: Maximum duration in seconds (None for infinite)
            enable_smoothing: Whether to apply action smoothing
        """
        try:
            # Setup robot and cameras
            self.setup_robot_interface()
            self.setup_image_client()
            
            # Wait for image client to receive first frame
            logging.info("Waiting for camera feed...")
            time.sleep(2.0)
            
            # Get initial observation to check if everything is working
            test_obs, test_state = self.get_observation()
            logging.info(f"Image shape: {test_obs['observation.images.cam_left_high'].shape}")
            logging.info(f"State shape: {test_state.shape}")
            
            # Ask for user confirmation
            logging.warning("=" * 80)
            logging.warning("SAFETY CHECK")
            logging.warning("=" * 80)
            logging.warning("1. Ensure robot is in a safe starting configuration")
            logging.warning("2. Ensure there are no obstacles in the workspace")
            logging.warning("3. Be ready to press emergency stop if needed")
            logging.warning("4. Press Ctrl+C to stop the deployment at any time")
            logging.warning("=" * 80)
            
            user_input = input("\nEnter 's' to start deployment, 'q' to quit: ").strip().lower()
            
            if user_input != 's':
                logging.info("Deployment cancelled by user")
                return
            
            # Reset policy
            self.policy.reset()
            self.previous_action = None
            
            # Main control loop
            logging.info("=" * 80)
            logging.info(f"Starting deployment at {frequency} Hz")
            logging.info("Press Ctrl+C to stop")
            logging.info("=" * 80)
            
            start_time = time.time()
            step_count = 0
            
            with torch.no_grad():
                while True:
                    loop_start = time.perf_counter()
                    
                    # Check duration limit
                    if duration and (time.time() - start_time) > duration:
                        logging.info(f"Reached duration limit of {duration}s")
                        break
                    
                    # 1. Get observation
                    observation, current_state = self.get_observation()
                    
                    # 2. Add batch dimension and move to device
                    batch = {
                        k: v.unsqueeze(0).to(self.device) 
                        for k, v in observation.items() 
                        if isinstance(v, torch.Tensor)
                    }
                    
                    # 3. Get action from policy
                    action = self.policy.select_action(batch)
                    action_np = action.cpu().numpy().squeeze()
                    
                    # 4. Apply safety smoothing
                    if enable_smoothing:
                        action_np = self.smooth_action(action_np)
                    
                    # 5. Execute action
                    self.execute_action(action_np)
                    
                    # 6. Logging (every 10 steps)
                    if step_count % 10 == 0:
                        logging.info(f"Step {step_count} | "
                                   f"State range: [{current_state.min():.3f}, {current_state.max():.3f}] | "
                                   f"Action range: [{action_np.min():.3f}, {action_np.max():.3f}]")
                    
                    step_count += 1
                    
                    # 7. Maintain frequency
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = max(0, (1.0 / frequency) - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        logging.warning(f"Loop took {elapsed*1000:.1f}ms (target: {1000/frequency:.1f}ms)")
                        
        except KeyboardInterrupt:
            logging.info("\n" + "=" * 80)
            logging.info("Emergency stop triggered by user")
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"Error during deployment: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        
        # Clean up shared memory
        if hasattr(self, 'img_shm'):
            try:
                self.img_shm.close()
                self.img_shm.unlink()
            except Exception as e:
                logging.warning(f"Error cleaning up shared memory: {e}")
        
        logging.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Deploy trained policy on real Unitree G1 robot")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., outputs/train/g1_diffusion_baseline/checkpoints/best)"
    )
    
    parser.add_argument(
        "--frequency",
        type=int,
        default=10,
        help="Control loop frequency in Hz (default: 10)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum deployment duration in seconds (default: infinite)"
    )
    
    parser.add_argument(
        "--arm-type",
        type=str,
        default="G1_29",
        choices=["G1_29", "G1_23"],
        help="Robot arm type (default: G1_29)"
    )
    
    parser.add_argument(
        "--motion-mode",
        action="store_true",
        help="Use motion mode for robot control"
    )
    
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode (for testing)"
    )
    
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable action smoothing (not recommended)"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    logging.info("=" * 80)
    logging.info("Unitree G1 Real Robot Deployment")
    logging.info("=" * 80)
    logging.info(f"Checkpoint:      {checkpoint_path}")
    logging.info(f"Frequency:       {args.frequency} Hz")
    logging.info(f"Duration:        {args.duration if args.duration else 'Infinite'}")
    logging.info(f"Arm Type:        {args.arm_type}")
    logging.info(f"Motion Mode:     {args.motion_mode}")
    logging.info(f"Simulation:      {args.simulation}")
    logging.info(f"Action Smoothing: {not args.no_smoothing}")
    logging.info(f"Device:          {DEVICE}")
    logging.info("=" * 80)
    
    # Create deployment manager
    deployer = RealRobotDeployment(
        checkpoint_path=checkpoint_path,
        arm_type=args.arm_type,
        motion_mode=args.motion_mode,
        simulation_mode=args.simulation
    )
    
    # Run deployment
    deployer.run(
        frequency=args.frequency,
        duration=args.duration,
        enable_smoothing=not args.no_smoothing
    )
    
    logging.info("Deployment complete")


if __name__ == "__main__":
    main()

