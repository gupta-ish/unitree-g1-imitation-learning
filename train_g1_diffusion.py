"""
Training script for Diffusion Policy on Unitree G1 Dataset.

Usage:
    python train_g1_diffusion.py [--config CONFIG_NAME]
    
Available configs:
    - baseline: Standard training (100k steps, lr=1e-4)
    - improved: Longer training with better hyperparameters (250k steps, lr=5e-5, larger batch)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ========================
# Configuration Presets
# ========================

CONFIGS = {
    "baseline": {
        "name": "baseline",
        "batch_size": 8,
        "num_workers": 4,
        "learning_rate": 1e-4,
        "training_steps": 100_000,
        "log_freq": 100,
        "save_freq": 10_000,
        "output_dir": "outputs/train/g1_diffusion_baseline",
        "weight_decay": 1e-6,
        "gradient_clip": 10.0,
        "warmup_steps": 500,
    },
    "improved": {
        "name": "improved",
        "batch_size": 16,  # Larger batch for more stable gradients
        "num_workers": 6,
        "learning_rate": 5e-5,  # Lower learning rate for finer optimization
        "training_steps": 250_000,  # 2.5x more training steps
        "log_freq": 100,
        "save_freq": 10_000,
        "output_dir": "outputs/train/g1_diffusion_improved",
        "weight_decay": 5e-6,  # Slightly higher weight decay for better generalization
        "gradient_clip": 5.0,  # Tighter gradient clipping
        "warmup_steps": 1000,  # Longer warmup for stability
    },
}

# Dataset configuration
DATASET_REPO_ID = "your_name/g1_dataset"
DATASET_ROOT = Path("/home/ishitagupta/zenavatar/lerobot_datasets/your_name/g1_dataset")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy configuration (same for all configs)
N_OBS_STEPS = 2  # Number of observation frames to use
HORIZON = 16  # Number of action steps to predict
N_ACTION_STEPS = 8  # Number of actions to execute per policy call

# Global config (will be set based on argument)
CONFIG = None
OUTPUT_DIR = None
CHECKPOINT_DIR = None

# ========================
# Setup
# ========================

def setup_logging(config_name):
    """Setup logging with config-specific log file."""
    log_file = f'train_g1_diffusion_{config_name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True  # Override any existing config
    )
    logging.info(f"Logging to: {log_file}")


def setup_directories():
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Checkpoint directory: {CHECKPOINT_DIR}")


def load_dataset(delta_timestamps=None):
    """Load the LeRobot dataset with proper configuration."""
    logging.info(f"Loading dataset: {DATASET_REPO_ID}")
    logging.info(f"Dataset root: {DATASET_ROOT}")
    
    # Load dataset metadata to get FPS and features
    # Use 'pyav' video backend for better compatibility
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        video_backend="pyav",  # Use pyav instead of torchcodec
        delta_timestamps=delta_timestamps,  # Pass delta_timestamps during initialization
    )
    
    logging.info(f"Dataset loaded successfully")
    logging.info(f"Number of episodes: {dataset.num_episodes}")
    logging.info(f"Number of frames: {dataset.num_frames}")
    logging.info(f"FPS: {dataset.fps}")
    
    return dataset


def create_policy_config(dataset):
    """Create policy configuration from dataset metadata."""
    logging.info("Creating policy configuration...")
    
    # Get features from dataset metadata
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    logging.info(f"Input features: {list(input_features.keys())}")
    logging.info(f"Output features: {list(output_features.keys())}")
    
    # Create policy configuration
    config = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=N_OBS_STEPS,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
    )
    
    return config


def create_policy(config, dataset):
    """Create and configure the diffusion policy."""
    logging.info("Creating diffusion policy...")
    
    # Initialize policy with dataset statistics for normalization
    policy = DiffusionPolicy(config, dataset_stats=dataset.meta.stats)
    policy.train()
    policy.to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {num_params:,}")
    logging.info(f"Trainable parameters: {num_trainable_params:,}")
    
    return policy


def create_dataloader(dataset):
    """Create dataloader."""
    logging.info("Creating dataloader...")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
        drop_last=True,
    )
    
    logging.info(f"Dataloader created with batch_size={CONFIG['batch_size']}, num_workers={CONFIG['num_workers']}")
    
    return dataloader


def get_delta_timestamps(config, fps):
    """Create delta timestamps dict for dataset loading."""
    delta_timestamps = {
        "observation.state": [i / fps for i in config.observation_delta_indices],
        "action": [i / fps for i in config.action_delta_indices],
    }
    return delta_timestamps


def train():
    """Main training loop."""
    # Setup
    setup_directories()
    
    # Check device
    logging.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    
    # Load dataset first to get metadata (without delta_timestamps)
    logging.info("Loading dataset metadata...")
    dataset_temp = load_dataset()
    fps = dataset_temp.fps
    
    # Create policy config
    config = create_policy_config(dataset_temp)
    
    # Create delta timestamps based on config
    delta_timestamps = get_delta_timestamps(config, fps)
    
    # Add delta timestamps for all image observations
    for key in dataset_temp.meta.features.keys():
        if key.startswith("observation.images."):
            delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]
    
    logging.info(f"Delta timestamps: {delta_timestamps}")
    
    # Now reload dataset WITH delta timestamps for temporal data
    logging.info("Reloading dataset with temporal configuration...")
    dataset = load_dataset(delta_timestamps=delta_timestamps)
    
    # Create policy and dataloader
    policy = create_policy(config, dataset)
    dataloader = create_dataloader(dataset)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Create learning rate scheduler with warmup
    # Linear warmup followed by cosine annealing
    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            # Linear warmup
            return step / CONFIG["warmup_steps"]
        else:
            # Cosine annealing after warmup
            progress = (step - CONFIG["warmup_steps"]) / (CONFIG["training_steps"] - CONFIG["warmup_steps"])
            return 0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    logging.info("=" * 80)
    logging.info("Starting training")
    logging.info(f"Configuration: {CONFIG['name']}")
    logging.info(f"Training steps: {CONFIG['training_steps']}")
    logging.info(f"Learning rate: {CONFIG['learning_rate']}")
    logging.info(f"Batch size: {CONFIG['batch_size']}")
    logging.info(f"Weight decay: {CONFIG['weight_decay']}")
    logging.info(f"Gradient clip: {CONFIG['gradient_clip']}")
    logging.info(f"Warmup steps: {CONFIG['warmup_steps']}")
    logging.info("=" * 80)
    
    # Training loop
    step = 0
    epoch = 0
    best_loss = float('inf')
    running_loss = 0.0
    
    dataloader_iter = iter(dataloader)
    
    progress_bar = tqdm(total=CONFIG["training_steps"], desc=f"Training ({CONFIG['name']})")
    
    while step < CONFIG["training_steps"]:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            epoch += 1
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            logging.info(f"Completed epoch {epoch}")
        
        # Move batch to device
        batch = {
            k: v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        # Add action_is_pad if missing (required by diffusion policy)
        if "action_is_pad" not in batch:
            # Create a padding mask (all False = no padding)
            batch_size, action_horizon = batch["action"].shape[0], batch["action"].shape[1]
            batch["action_is_pad"] = torch.zeros(batch_size, action_horizon, dtype=torch.bool, device=DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        loss, _ = policy.forward(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=CONFIG["gradient_clip"])
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Logging
        running_loss += loss.item()
        
        if step % CONFIG["log_freq"] == 0 and step > 0:
            avg_loss = running_loss / CONFIG["log_freq"]
            current_lr = scheduler.get_last_lr()[0]
            
            logging.info(
                f"Step {step}/{CONFIG['training_steps']} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
            running_loss = 0.0
        
        # Save checkpoint
        if step % CONFIG["save_freq"] == 0 and step > 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{step}"
            checkpoint_path.mkdir(exist_ok=True)
            
            logging.info(f"Saving checkpoint to {checkpoint_path}")
            policy.save_pretrained(checkpoint_path)
            
            # Save training state with hyperparameters
            training_state = {
                'step': step,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'config': CONFIG,  # Save hyperparameters
                'policy_config': {
                    'n_obs_steps': N_OBS_STEPS,
                    'horizon': HORIZON,
                    'n_action_steps': N_ACTION_STEPS,
                },
            }
            torch.save(training_state, checkpoint_path / "training_state.pt")
            
            # Also save hyperparameters as JSON for easy viewing
            with open(checkpoint_path / "hyperparameters.json", 'w') as f:
                json.dump({
                    'training_config': CONFIG,
                    'policy_config': {
                        'n_obs_steps': N_OBS_STEPS,
                        'horizon': HORIZON,
                        'n_action_steps': N_ACTION_STEPS,
                    },
                    'training_step': step,
                    'training_epoch': epoch,
                    'loss': loss.item(),
                }, f, indent=2)
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_path = CHECKPOINT_DIR / "best"
                best_path.mkdir(exist_ok=True)
                logging.info(f"New best loss: {best_loss:.4f}, saving to {best_path}")
                policy.save_pretrained(best_path)
                
                # Save best model hyperparameters too
                torch.save(training_state, best_path / "training_state.pt")
                with open(best_path / "hyperparameters.json", 'w') as f:
                    json.dump({
                        'training_config': CONFIG,
                        'policy_config': {
                            'n_obs_steps': N_OBS_STEPS,
                            'horizon': HORIZON,
                            'n_action_steps': N_ACTION_STEPS,
                        },
                        'training_step': step,
                        'training_epoch': epoch,
                        'loss': loss.item(),
                        'best_checkpoint': True,
                    }, f, indent=2)
        
        step += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Save final checkpoint
    final_path = CHECKPOINT_DIR / "final"
    final_path.mkdir(exist_ok=True)
    logging.info(f"Training completed! Saving final model to {final_path}")
    policy.save_pretrained(final_path)
    
    # Save final training state with hyperparameters
    final_training_state = {
        'step': step,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
        'config': CONFIG,
        'policy_config': {
            'n_obs_steps': N_OBS_STEPS,
            'horizon': HORIZON,
            'n_action_steps': N_ACTION_STEPS,
        },
    }
    torch.save(final_training_state, final_path / "training_state.pt")
    
    # Save final hyperparameters as JSON
    with open(final_path / "hyperparameters.json", 'w') as f:
        json.dump({
            'training_config': CONFIG,
            'policy_config': {
                'n_obs_steps': N_OBS_STEPS,
                'horizon': HORIZON,
                'n_action_steps': N_ACTION_STEPS,
            },
            'training_step': step,
            'training_epoch': epoch,
            'loss': loss.item(),
            'final_checkpoint': True,
        }, f, indent=2)
    
    logging.info("=" * 80)
    logging.info("Training finished successfully!")
    logging.info(f"Final model saved to: {final_path}")
    logging.info(f"Configuration: {CONFIG['name']}")
    logging.info("=" * 80)


def main():
    """Main entry point with argument parsing."""
    global CONFIG, OUTPUT_DIR, CHECKPOINT_DIR
    
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy on Unitree G1 Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available configurations:
  baseline  - Standard training (100k steps, lr=1e-4, batch=8)
  improved  - Longer training with better hyperparameters (250k steps, lr=5e-5, batch=16)
  
Example:
  python train_g1_diffusion.py --config improved
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="baseline",
        choices=list(CONFIGS.keys()),
        help="Training configuration to use"
    )
    
    args = parser.parse_args()
    
    # Set global config
    CONFIG = CONFIGS[args.config]
    OUTPUT_DIR = Path(CONFIG["output_dir"])
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    
    # Setup logging
    setup_logging(CONFIG["name"])
    
    logging.info("=" * 80)
    logging.info("Unitree G1 Diffusion Policy Training")
    logging.info("=" * 80)
    logging.info(f"Selected configuration: {CONFIG['name']}")
    
    try:
        train()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

