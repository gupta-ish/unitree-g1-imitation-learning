"""
Evaluation script for trained Diffusion Policy on Unitree G1 Dataset.

This script allows you to:
1. Visualize policy predictions vs ground truth actions
2. Compute evaluation metrics (MSE, MAE, etc.)
3. Generate video rollouts showing what the robot would do
4. Analyze per-joint performance

Usage:
    python evaluate_g1_policy.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Dataset configuration
DATASET_REPO_ID = "your_name/g1_dataset"
DATASET_ROOT = Path("/home/ishitagupta/zenavatar/lerobot_datasets/your_name/g1_dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyEvaluator:
    """Evaluator for trained diffusion policy."""
    
    def __init__(self, checkpoint_path: Path, dataset: LeRobotDataset):
        self.checkpoint_path = Path(checkpoint_path).resolve()  # Convert to absolute path
        self.dataset = dataset
        self.device = DEVICE
        
        # Load policy
        logging.info(f"Loading policy from {self.checkpoint_path}")
        self.policy = DiffusionPolicy.from_pretrained(str(self.checkpoint_path))
        self.policy.eval()
        self.policy.to(self.device)
        logging.info("Policy loaded successfully")
        
        # Get motor names from dataset
        self.motor_names = dataset.meta.features["action"]["names"][0]
        self.num_motors = len(self.motor_names)
        
    def evaluate_episode(self, episode_idx: int):
        """Evaluate policy on a single episode."""
        logging.info(f"Evaluating episode {episode_idx}")
        
        # Get all frames from this episode
        episode_data = self.dataset.meta.episodes[episode_idx]
        episode_length = episode_data["length"]
        
        # Collect predictions and ground truth
        predicted_actions = []
        ground_truth_actions = []
        observations = []
        
        # Get episode start index
        start_idx = sum(self.dataset.meta.episodes[i]["length"] for i in range(episode_idx))
        
        self.policy.reset()
        
        with torch.no_grad():
            for frame_idx in tqdm(range(episode_length), desc=f"Episode {episode_idx}"):
                # Get frame from dataset
                frame = self.dataset[start_idx + frame_idx]
                
                # Prepare batch (add batch dimension)
                batch = {k: v.unsqueeze(0).to(self.device) for k, v in frame.items() if isinstance(v, torch.Tensor)}
                
                # Get policy prediction
                action = self.policy.select_action(batch)
                
                # Store results
                # Squeeze to remove extra dimensions and ensure consistent shape
                pred_action = action.cpu().numpy().squeeze()
                gt_action = frame["action"][0].cpu().numpy().squeeze() if frame["action"].ndim > 1 else frame["action"].cpu().numpy()
                
                predicted_actions.append(pred_action)
                ground_truth_actions.append(gt_action)
                
                # Store observation for visualization
                if "observation.images.cam_left_high" in frame:
                    obs_img = frame["observation.images.cam_left_high"]
                    if obs_img.dim() == 4:  # [T, C, H, W]
                        obs_img = obs_img[-1]  # Take most recent
                    observations.append(obs_img.cpu().numpy())
        
        predicted_actions = np.array(predicted_actions)
        ground_truth_actions = np.array(ground_truth_actions)
        
        return {
            "predicted_actions": predicted_actions,
            "ground_truth_actions": ground_truth_actions,
            "observations": observations,
            "episode_idx": episode_idx,
            "episode_length": episode_length,
        }
    
    def compute_metrics(self, results):
        """Compute evaluation metrics."""
        pred = results["predicted_actions"]
        gt = results["ground_truth_actions"]
        
        # Overall metrics
        mse = np.mean((pred - gt) ** 2)
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(mse)
        
        # Per-joint metrics
        per_joint_mse = np.mean((pred - gt) ** 2, axis=0)
        per_joint_mae = np.mean(np.abs(pred - gt), axis=0)
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "per_joint_mse": per_joint_mse.tolist(),
            "per_joint_mae": per_joint_mae.tolist(),
        }
        
        return metrics
    
    def plot_action_comparison(self, results, output_path: Path):
        """Plot predicted vs ground truth actions."""
        pred = results["predicted_actions"]
        gt = results["ground_truth_actions"]
        episode_idx = results["episode_idx"]
        
        # Create figure with subplots for each joint
        num_joints = pred.shape[1]
        cols = 4
        rows = (num_joints + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
        fig.suptitle(f"Episode {episode_idx}: Predicted vs Ground Truth Actions", fontsize=16)
        
        axes = axes.flatten() if num_joints > 1 else [axes]
        
        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            
            timesteps = np.arange(len(pred))
            ax.plot(timesteps, gt[:, joint_idx], label='Ground Truth', linewidth=2, alpha=0.7)
            ax.plot(timesteps, pred[:, joint_idx], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
            
            motor_name = self.motor_names[joint_idx] if joint_idx < len(self.motor_names) else f"Joint {joint_idx}"
            ax.set_title(motor_name, fontsize=10)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Action Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(num_joints, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Action comparison plot saved to {output_path}")
    
    def plot_error_distribution(self, results, output_path: Path):
        """Plot error distribution across joints."""
        pred = results["predicted_actions"]
        gt = results["ground_truth_actions"]
        errors = pred - gt
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Error over time for all joints
        ax = axes[0]
        timesteps = np.arange(len(errors))
        for joint_idx in range(errors.shape[1]):
            motor_name = self.motor_names[joint_idx] if joint_idx < len(self.motor_names) else f"Joint {joint_idx}"
            ax.plot(timesteps, errors[:, joint_idx], label=motor_name, alpha=0.6)
        ax.set_title('Prediction Error Over Time')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Error')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Box plot of errors per joint
        ax = axes[1]
        ax.boxplot([errors[:, i] for i in range(errors.shape[1])], labels=self.motor_names)
        ax.set_title('Error Distribution by Joint')
        ax.set_xlabel('Joint')
        ax.set_ylabel('Error')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Error distribution plot saved to {output_path}")
    
    def create_video_rollout(self, results, output_path: Path, fps=10):
        """Create video showing observations with predicted actions overlaid."""
        if not results["observations"]:
            logging.warning("No observations available for video creation")
            return
        
        observations = results["observations"]
        pred = results["predicted_actions"]
        gt = results["ground_truth_actions"]
        
        # Get frame dimensions
        first_frame = observations[0]
        if first_frame.shape[0] == 3:  # CHW format
            first_frame = np.transpose(first_frame, (1, 2, 0))
        
        h, w = first_frame.shape[:2]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create video writer with a more compatible codec
        # Try different codecs for better compatibility
        fourcc_options = [
            ('avc1', 'H.264'),  # Best quality and compatibility
            ('XVID', 'XVID'),   # Good fallback
            ('MJPG', 'MJPEG'),  # Motion JPEG - very compatible
            ('mp4v', 'MPEG-4'), # Last resort
        ]
        
        video_writer = None
        used_codec = None
        
        for codec, name in fourcc_options:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))
            if video_writer.isOpened():
                used_codec = name
                logging.info(f"Using {name} codec for video")
                break
            video_writer.release()
        
        if not video_writer or not video_writer.isOpened():
            logging.error("Failed to create video writer with any codec")
            return
        
        for frame_idx in range(len(observations)):
            # Get observation image
            obs = observations[frame_idx]
            if obs.shape[0] == 3:  # CHW -> HWC
                obs = np.transpose(obs, (1, 2, 0))
            
            # Normalize to 0-255 if needed
            if obs.max() <= 1.0:
                obs = (obs * 255).astype(np.uint8)
            else:
                obs = obs.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            
            # Create action visualization panel
            action_panel = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Add text with action values (show first 6 joints)
            y_offset = 30
            num_joints_to_show = min(8, len(pred[frame_idx]))
            
            cv2.putText(action_panel, "Action Comparison", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            cv2.putText(action_panel, "Joint | Pred | GT | Error", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
            
            for i in range(num_joints_to_show):
                pred_val = pred[frame_idx, i]
                gt_val = gt[frame_idx, i]
                error = pred_val - gt_val
                
                motor_name = self.motor_names[i] if i < len(self.motor_names) else f"J{i}"
                text = f"{motor_name[:8]:8s} {pred_val:6.3f} {gt_val:6.3f} {error:6.3f}"
                
                # Color code by error magnitude
                color = (0, 255, 0) if abs(error) < 0.1 else (0, 255, 255) if abs(error) < 0.3 else (0, 0, 255)
                
                cv2.putText(action_panel, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 20
            
            # Add frame number
            cv2.putText(action_panel, f"Frame: {frame_idx}/{len(observations)}", 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Combine observation and action panel
            combined_frame = np.hstack([obs_bgr, action_panel])
            success = video_writer.write(combined_frame)
            if not success and frame_idx == 0:
                logging.warning(f"Failed to write frame {frame_idx}")
        
        video_writer.release()
        
        # Verify the video was created successfully
        if output_path.exists():
            file_size = output_path.stat().st_size
            logging.info(f"Video rollout saved to {output_path}")
            logging.info(f"Video file size: {file_size / (1024*1024):.2f} MB")
            logging.info(f"Codec used: {used_codec}")
        else:
            logging.error(f"Video file was not created at {output_path}")
    
    def generate_report(self, results, metrics, output_dir: Path):
        """Generate a comprehensive evaluation report."""
        report_path = output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNITREE G1 DIFFUSION POLICY EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Episode: {results['episode_idx']}\n")
            f.write(f"Episode Length: {results['episode_length']} frames\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Squared Error (MSE):  {metrics['mse']:.6f}\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}\n")
            f.write(f"Root Mean Squared Error:   {metrics['rmse']:.6f}\n\n")
            
            f.write("PER-JOINT METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Joint Name':<20} {'MSE':>12} {'MAE':>12}\n")
            f.write("-" * 80 + "\n")
            
            for i, motor_name in enumerate(self.motor_names):
                if i < len(metrics['per_joint_mse']):
                    mse = metrics['per_joint_mse'][i]
                    mae = metrics['per_joint_mae'][i]
                    f.write(f"{motor_name:<20} {mse:>12.6f} {mae:>12.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logging.info(f"Evaluation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Diffusion Policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., outputs/train/g1_diffusion/checkpoints/best)"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to evaluate (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--create-video",
        action="store_true",
        help="Create video rollout (can be slow)"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info("Unitree G1 Policy Evaluation")
    logging.info("=" * 80)
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Device: {DEVICE}")
    
    # Load dataset
    logging.info("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        video_backend="pyav",
    )
    logging.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    
    # Create evaluator
    evaluator = PolicyEvaluator(checkpoint_path, dataset)
    
    # Evaluate episode
    results = evaluator.evaluate_episode(args.episode)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(results)
    
    # Log metrics
    logging.info("\nEvaluation Metrics:")
    logging.info(f"  MSE:  {metrics['mse']:.6f}")
    logging.info(f"  MAE:  {metrics['mae']:.6f}")
    logging.info(f"  RMSE: {metrics['rmse']:.6f}")
    
    # Generate visualizations
    logging.info("\nGenerating visualizations...")
    
    evaluator.plot_action_comparison(
        results,
        output_dir / f"action_comparison_ep{args.episode}.png"
    )
    
    evaluator.plot_error_distribution(
        results,
        output_dir / f"error_distribution_ep{args.episode}.png"
    )
    
    if args.create_video:
        evaluator.create_video_rollout(
            results,
            output_dir / f"rollout_ep{args.episode}.mp4"
        )
    
    # Generate report
    evaluator.generate_report(results, metrics, output_dir)
    
    logging.info("\n" + "=" * 80)
    logging.info("Evaluation complete!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

