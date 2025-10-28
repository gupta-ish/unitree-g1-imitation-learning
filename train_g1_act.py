"""
Training script for Action Chunking Transformers (ACT) Policy on Unitree G1 Dataset.

ACT is a powerful transformer-based policy that predicts action chunks using temporal ensembling.
Paper: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
       https://huggingface.co/papers/2304.13705

Usage:
    python train_g1_act.py [--config CONFIG_NAME]
    
Available configs:
    - baseline: Standard ACT training (100k steps)
    - large: Larger model with more capacity (200k steps)
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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

# ========================
# Configuration Presets
# ========================

CONFIGS = {
    "baseline": {
        "name": "baseline",
        "batch_size": 8,
        "num_workers": 4,
        "learning_rate": 1e-5,
        "learning_rate_backbone": 1e-5,
        "training_steps": 100_000,
        "log_freq": 100,
        "save_freq": 10_000,
        "output_dir": "outputs/train/g1_act_baseline",
        "weight_decay": 1e-4,
        "gradient_clip": 10.0,
        "warmup_steps": 1000,
        # ACT-specific parameters
        "chunk_size": 100,  # Action chunk size
        "n_obs_steps": 1,  # Observation history
        "n_action_steps": 100,  # Execute all predicted actions
        "dim_model": 512,  # Transformer hidden dimension
        "n_heads": 8,  # Attention heads
        "n_encoder_layers": 4,  # Transformer encoder layers
        "n_decoder_layers": 1,  # Transformer decoder layers (ACT uses 1 due to original bug)
        "latent_dim": 32,  # VAE latent dimension
        "use_vae": True,  # Use variational objective
        "kl_weight": 10.0,  # KL divergence weight
    },
    "large": {
        "name": "large",
        "batch_size": 8,  # Keep same batch size
        "num_workers": 6,
        "learning_rate": 5e-6,  # Lower LR for larger model
        "learning_rate_backbone": 1e-6,  # Even lower for pretrained backbone
        "training_steps": 200_000,  # More steps for convergence
        "log_freq": 100,
        "save_freq": 10_000,
        "output_dir": "outputs/train/g1_act_large",
        "weight_decay": 1e-4,
        "gradient_clip": 10.0,
        "warmup_steps": 2000,  # Longer warmup
        # ACT-specific parameters - larger model
        "chunk_size": 100,
        "n_obs_steps": 1,
        "n_action_steps": 100,
        "dim_model": 1024,  # 2x larger hidden dimension
        "n_heads": 16,  # 2x more attention heads
        "n_encoder_layers": 6,  # More encoder layers
        "n_decoder_layers": 1,  # Keep at 1 (ACT design)
        "latent_dim": 64,  # Larger latent space
        "use_vae": True,
        "kl_weight": 10.0,
    },
}

# Dataset configuration
DATASET_REPO_ID = "your_name/g1_dataset"
DATASET_ROOT = Path("/home/ishitagupta/zenavatar/lerobot_datasets/your_name/g1_dataset")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global config (will be set based on argument)
CONFIG = None
OUTPUT_DIR = None
CHECKPOINT_DIR = None

# ========================
# Setup
# ========================

def setup_logging(config_name):
    """Setup logging with config-specific log file."""
    log_file = f'train_g1_act_{config_name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True
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
    
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        video_backend="pyav",
        delta_timestamps=delta_timestamps,
    )
    
    logging.info(f"Dataset loaded successfully")
    logging.info(f"Number of episodes: {dataset.num_episodes}")
    logging.info(f"Number of frames: {dataset.num_frames}")
    logging.info(f"FPS: {dataset.fps}")
    
    return dataset


def create_policy_config(dataset):
    """Create ACT policy configuration from dataset metadata."""
    logging.info("Creating ACT policy configuration...")
    
    # Get features from dataset metadata
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    logging.info(f"Input features: {list(input_features.keys())}")
    logging.info(f"Output features: {list(output_features.keys())}")
    
    # Create ACT policy configuration
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=CONFIG["n_obs_steps"],
        chunk_size=CONFIG["chunk_size"],
        n_action_steps=CONFIG["n_action_steps"],
        # Architecture parameters
        dim_model=CONFIG["dim_model"],
        n_heads=CONFIG["n_heads"],
        n_encoder_layers=CONFIG["n_encoder_layers"],
        n_decoder_layers=CONFIG["n_decoder_layers"],
        # VAE parameters
        use_vae=CONFIG["use_vae"],
        latent_dim=CONFIG["latent_dim"],
        kl_weight=CONFIG["kl_weight"],
        # Vision backbone
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        # Optimizer settings (will be used by the policy's get_optim_params)
        optimizer_lr=CONFIG["learning_rate"],
        optimizer_lr_backbone=CONFIG["learning_rate_backbone"],
        optimizer_weight_decay=CONFIG["weight_decay"],
    )
    
    logging.info(f"ACT Config:")
    logging.info(f"  - Chunk size: {config.chunk_size}")
    logging.info(f"  - Obs steps: {config.n_obs_steps}")
    logging.info(f"  - Action steps: {config.n_action_steps}")
    logging.info(f"  - Model dim: {config.dim_model}")
    logging.info(f"  - Attention heads: {config.n_heads}")
    logging.info(f"  - Encoder layers: {config.n_encoder_layers}")
    logging.info(f"  - Decoder layers: {config.n_decoder_layers}")
    logging.info(f"  - Use VAE: {config.use_vae}")
    logging.info(f"  - Latent dim: {config.latent_dim}")
    
    return config


def create_policy(config, dataset):
    """Create and configure the ACT policy."""
    logging.info("Creating ACT policy...")
    
    # Initialize policy with dataset statistics for normalization
    policy = ACTPolicy(config, dataset_stats=dataset.meta.stats)
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
    # For ACT with n_obs_steps=1, we only need current observation (index 0)
    # observation_delta_indices goes backwards: [0, -1, -2, ...] for n_obs_steps
    obs_indices = list(range(0, -config.n_obs_steps, -1))
    
    # For actions, we need chunk_size future actions: [0, 1, 2, ..., chunk_size-1]
    action_indices = list(range(config.chunk_size))
    
    delta_timestamps = {
        "observation.state": [i / fps for i in obs_indices],
        "action": [i / fps for i in action_indices],
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
    
    # Load dataset first to get metadata
    logging.info("Loading dataset metadata...")
    dataset_temp = load_dataset()
    fps = dataset_temp.fps
    
    # Create policy config
    config = create_policy_config(dataset_temp)
    
    # For ACT, we need delta_timestamps for action chunks
    # but n_obs_steps=1 means no temporal stacking for observations
    delta_timestamps = {}
    
    # Only actions need temporal dimension (for chunk_size)
    action_indices = list(range(config.chunk_size))
    delta_timestamps["action"] = [i / fps for i in action_indices]
    
    logging.info(f"Delta timestamps: {delta_timestamps}")
    logging.info(f"Note: ACT with n_obs_steps=1 uses single-frame observations (no temporal stacking)")
    
    # Reload dataset with temporal configuration
    logging.info("Reloading dataset with temporal configuration...")
    dataset = load_dataset(delta_timestamps=delta_timestamps)
    
    # Create policy and dataloader
    policy = create_policy(config, dataset)
    dataloader = create_dataloader(dataset)
    
    # Create optimizer with different learning rates for backbone and other params
    # ACT uses different learning rates for the pretrained vision backbone
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Create learning rate scheduler with warmup
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
    logging.info("Starting ACT Policy Training")
    logging.info(f"Configuration: {CONFIG['name']}")
    logging.info(f"Training steps: {CONFIG['training_steps']}")
    logging.info(f"Learning rate: {CONFIG['learning_rate']}")
    logging.info(f"LR (backbone): {CONFIG['learning_rate_backbone']}")
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
    running_l1_loss = 0.0
    running_kld_loss = 0.0
    
    dataloader_iter = iter(dataloader)
    
    progress_bar = tqdm(total=CONFIG["training_steps"], desc=f"Training ACT ({CONFIG['name']})")
    
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
        
        # Add action_is_pad if missing (required by ACT policy)
        if "action_is_pad" not in batch:
            # Create a padding mask (all False = no padding)
            batch_size, action_horizon = batch["action"].shape[0], batch["action"].shape[1]
            batch["action_is_pad"] = torch.zeros(batch_size, action_horizon, dtype=torch.bool, device=DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        loss, loss_dict = policy.forward(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=CONFIG["gradient_clip"])
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Logging
        running_loss += loss.item()
        running_l1_loss += loss_dict.get("l1_loss", 0.0)
        running_kld_loss += loss_dict.get("kld_loss", 0.0)
        
        if step % CONFIG["log_freq"] == 0 and step > 0:
            avg_loss = running_loss / CONFIG["log_freq"]
            avg_l1_loss = running_l1_loss / CONFIG["log_freq"]
            avg_kld_loss = running_kld_loss / CONFIG["log_freq"]
            current_lr = scheduler.get_last_lr()[0]
            
            log_msg = (
                f"Step {step}/{CONFIG['training_steps']} | "
                f"Loss: {avg_loss:.4f} | "
                f"L1: {avg_l1_loss:.4f}"
            )
            if CONFIG["use_vae"]:
                log_msg += f" | KLD: {avg_kld_loss:.4f}"
            log_msg += f" | LR: {current_lr:.2e}"
            
            logging.info(log_msg)
            
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "l1": f"{avg_l1_loss:.4f}",
                "kld": f"{avg_kld_loss:.4f}" if CONFIG["use_vae"] else "N/A",
                "lr": f"{current_lr:.2e}"
            })
            
            running_loss = 0.0
            running_l1_loss = 0.0
            running_kld_loss = 0.0
        
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
                'config': CONFIG,
                'policy_config': {
                    'chunk_size': CONFIG["chunk_size"],
                    'n_obs_steps': CONFIG["n_obs_steps"],
                    'n_action_steps': CONFIG["n_action_steps"],
                },
            }
            torch.save(training_state, checkpoint_path / "training_state.pt")
            
            # Save hyperparameters as JSON
            with open(checkpoint_path / "hyperparameters.json", 'w') as f:
                json.dump({
                    'training_config': CONFIG,
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
                
                torch.save(training_state, best_path / "training_state.pt")
                with open(best_path / "hyperparameters.json", 'w') as f:
                    json.dump({
                        'training_config': CONFIG,
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
    
    final_training_state = {
        'step': step,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
        'config': CONFIG,
    }
    torch.save(final_training_state, final_path / "training_state.pt")
    
    with open(final_path / "hyperparameters.json", 'w') as f:
        json.dump({
            'training_config': CONFIG,
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
        description="Train Action Chunking Transformers (ACT) Policy on Unitree G1 Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available configurations:
  baseline  - Standard ACT training (100k steps, dim=512, 4 encoder layers)
  large     - Larger ACT model (200k steps, dim=1024, 6 encoder layers)
  
Example:
  python train_g1_act.py --config baseline
  python train_g1_act.py --config large
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
    logging.info("Unitree G1 ACT Policy Training")
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

