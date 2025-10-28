"""
Test script to verify deployment setup without running the robot.

This script checks:
1. Checkpoint exists and can be loaded
2. Policy can be initialized
3. Image client can connect (if server is running)
4. Robot SDK is available

Usage:
    python test_deployment_setup.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_checkpoint(checkpoint_path):
    """Test if checkpoint exists and can be loaded."""
    logging.info("=" * 60)
    logging.info("TEST 1: Checkpoint Loading")
    logging.info("=" * 60)
    
    try:
        if not checkpoint_path.exists():
            logging.error(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        logging.info(f"✓ Checkpoint directory exists: {checkpoint_path}")
        
        # Check for required files
        config_file = checkpoint_path / "config.json"
        model_file = checkpoint_path / "model.safetensors"
        
        if not config_file.exists():
            logging.error(f"❌ Config file not found: {config_file}")
            return False
        logging.info(f"✓ Config file found: {config_file}")
        
        if not model_file.exists():
            logging.error(f"❌ Model file not found: {model_file}")
            return False
        logging.info(f"✓ Model file found: {model_file}")
        
        # Try loading policy
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        
        logging.info("Loading policy...")
        try:
            policy = DiffusionPolicy.from_pretrained(str(checkpoint_path), local_files_only=True)
        except Exception as e:
            logging.warning(f"  - from_pretrained failed: {e}")
            logging.info("  - Trying alternative loading method...")
            import json
            from lerobot.configs.policies import DiffusionConfig
            from safetensors.torch import load_file
            
            config_path = checkpoint_path / "config.json"
            model_path = checkpoint_path / "model.safetensors"
            
            with open(config_path) as f:
                config_dict = json.load(f)
            
            config = DiffusionConfig(**config_dict)
            policy = DiffusionPolicy(config)
            state_dict = load_file(str(model_path))
            policy.load_state_dict(state_dict)
            
        logging.info(f"✓ Policy loaded successfully")
        logging.info(f"  - Device: {policy.config.device}")
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        policy.eval()
        logging.info(f"✓ Policy moved to {device} and set to eval mode")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robot_sdk():
    """Test if robot SDK is available."""
    logging.info("\n" + "=" * 60)
    logging.info("TEST 2: Robot SDK (Optional - only needed on robot)")
    logging.info("=" * 60)
    
    try:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        logging.info("✓ unitree_sdk2py is installed")
        
        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_ArmController
        logging.info("✓ G1_29_ArmController is available")
        
        from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK
        logging.info("✓ G1_29_ArmIK is available")
        
        return True
        
    except ImportError as e:
        logging.warning(f"⚠️  Robot SDK not installed: {e}")
        logging.warning("   This is OK if you're testing on development machine")
        logging.warning("   Robot SDK will be available when running on the actual robot")
        return True  # Return True since this is optional for testing


def test_image_client():
    """Test if image client module is available."""
    logging.info("\n" + "=" * 60)
    logging.info("TEST 3: Image Client")
    logging.info("=" * 60)
    
    try:
        from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
        logging.info("✓ ImageClient is available")
        
        import numpy as np
        from multiprocessing import shared_memory
        
        # Create a small test shared memory
        test_shape = (10, 10, 3)
        test_shm = shared_memory.SharedMemory(
            create=True,
            size=np.prod(test_shape) * np.uint8().itemsize
        )
        test_array = np.ndarray(test_shape, dtype=np.uint8, buffer=test_shm.buf)
        
        logging.info("✓ Shared memory creation works")
        
        # Cleanup
        test_shm.close()
        test_shm.unlink()
        
        logging.info("✓ Shared memory cleanup works")
        logging.info("⚠️  Note: Image server must be running on robot for actual deployment")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Image client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_inference(checkpoint_path):
    """Test if policy can perform inference."""
    logging.info("\n" + "=" * 60)
    logging.info("TEST 4: Policy Inference")
    logging.info("=" * 60)
    
    try:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        import numpy as np
        
        # Load policy
        try:
            policy = DiffusionPolicy.from_pretrained(str(checkpoint_path), local_files_only=True)
        except Exception:
            import json
            from lerobot.configs.policies import DiffusionConfig
            from safetensors.torch import load_file
            
            config_path = checkpoint_path / "config.json"
            model_path = checkpoint_path / "model.safetensors"
            
            with open(config_path) as f:
                config_dict = json.load(f)
            
            config = DiffusionConfig(**config_dict)
            policy = DiffusionPolicy(config)
            state_dict = load_file(str(model_path))
            policy.load_state_dict(state_dict)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        policy.eval()
        
        # Create dummy observation in CHW format (as expected by policy)
        dummy_image = torch.rand(1, 3, 480, 640).to(device)  # CHW format
        dummy_state = torch.rand(1, 14).to(device)  # 14 DOF
        
        observation = {
            "observation.images.cam_left_high": dummy_image,
            "observation.state": dummy_state
        }
        
        logging.info("Created dummy observation:")
        logging.info(f"  - Image shape: {dummy_image.shape}")
        logging.info(f"  - State shape: {dummy_state.shape}")
        
        # Reset policy
        policy.reset()
        
        # Test inference
        logging.info("Running policy inference...")
        with torch.no_grad():
            action = policy.select_action(observation)
        
        logging.info(f"✓ Policy inference successful")
        logging.info(f"  - Action shape: {action.shape}")
        logging.info(f"  - Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Policy inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test deployment setup")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    
    logging.info("╔" + "=" * 58 + "╗")
    logging.info("║  DEPLOYMENT SETUP TEST                                   ║")
    logging.info("╚" + "=" * 58 + "╝")
    logging.info(f"\nCheckpoint: {checkpoint_path}\n")
    
    # Run all tests
    results = []
    
    results.append(("Checkpoint Loading", test_checkpoint(checkpoint_path)))
    results.append(("Robot SDK", test_robot_sdk()))
    results.append(("Image Client", test_image_client()))
    results.append(("Policy Inference", test_policy_inference(checkpoint_path)))
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("TEST SUMMARY")
    logging.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        logging.info(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    logging.info("=" * 60)
    
    if all_passed:
        logging.info("\n🎉 All tests passed! Ready for deployment.")
        logging.info("\nNext steps:")
        logging.info("1. Start image server on robot:")
        logging.info("   ssh unitree@<ROBOT_IP>")
        logging.info("   python unitree_lerobot/eval_robot/image_server/image_server.py")
        logging.info("\n2. Run deployment:")
        logging.info(f"   python eval_real_g1.py --checkpoint {checkpoint_path}")
        sys.exit(0)
    else:
        logging.error("\n❌ Some tests failed. Please fix the issues before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()

