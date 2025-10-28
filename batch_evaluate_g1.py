"""
Batch evaluation script for testing policy on multiple episodes.

Usage:
    python batch_evaluate_g1.py --checkpoint outputs/train/g1_diffusion_baseline/checkpoints/best --num-episodes 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate_g1_policy import PolicyEvaluator
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

DATASET_REPO_ID = "your_name/g1_dataset"
DATASET_ROOT = Path("/home/ishitagupta/zenavatar/lerobot_datasets/your_name/g1_dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate policy on multiple episodes")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Starting episode index"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch_evaluation",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info("Batch Evaluation")
    logging.info("=" * 80)
    
    # Load dataset
    logging.info("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        video_backend="pyav",
    )
    
    # Create evaluator
    evaluator = PolicyEvaluator(checkpoint_path, dataset)
    
    # Evaluate multiple episodes
    all_metrics = []
    
    end_episode = min(args.start_episode + args.num_episodes, dataset.num_episodes)
    
    for ep_idx in range(args.start_episode, end_episode):
        logging.info(f"\nEvaluating episode {ep_idx}/{end_episode}")
        
        try:
            results = evaluator.evaluate_episode(ep_idx)
            metrics = evaluator.compute_metrics(results)
            metrics['episode_idx'] = ep_idx
            all_metrics.append(metrics)
            
        except Exception as e:
            logging.error(f"Error evaluating episode {ep_idx}: {e}")
            continue
    
    # Compute aggregate statistics
    logging.info("\n" + "=" * 80)
    logging.info("Aggregate Results")
    logging.info("=" * 80)
    
    mse_values = [m['mse'] for m in all_metrics]
    mae_values = [m['mae'] for m in all_metrics]
    rmse_values = [m['rmse'] for m in all_metrics]
    
    aggregate_stats = {
        "num_episodes_evaluated": len(all_metrics),
        "mse": {
            "mean": float(np.mean(mse_values)),
            "std": float(np.std(mse_values)),
            "min": float(np.min(mse_values)),
            "max": float(np.max(mse_values)),
        },
        "mae": {
            "mean": float(np.mean(mae_values)),
            "std": float(np.std(mae_values)),
            "min": float(np.min(mae_values)),
            "max": float(np.max(mae_values)),
        },
        "rmse": {
            "mean": float(np.mean(rmse_values)),
            "std": float(np.std(rmse_values)),
            "min": float(np.min(rmse_values)),
            "max": float(np.max(rmse_values)),
        },
    }
    
    # Per-joint aggregate statistics
    num_joints = len(all_metrics[0]['per_joint_mse'])
    per_joint_mse = np.array([m['per_joint_mse'] for m in all_metrics])
    per_joint_mae = np.array([m['per_joint_mae'] for m in all_metrics])
    
    aggregate_stats["per_joint"] = {
        "mse_mean": per_joint_mse.mean(axis=0).tolist(),
        "mse_std": per_joint_mse.std(axis=0).tolist(),
        "mae_mean": per_joint_mae.mean(axis=0).tolist(),
        "mae_std": per_joint_mae.std(axis=0).tolist(),
    }
    
    # Print results
    logging.info(f"\nEvaluated {len(all_metrics)} episodes")
    logging.info(f"\nMSE:  {aggregate_stats['mse']['mean']:.6f} ± {aggregate_stats['mse']['std']:.6f}")
    logging.info(f"MAE:  {aggregate_stats['mae']['mean']:.6f} ± {aggregate_stats['mae']['std']:.6f}")
    logging.info(f"RMSE: {aggregate_stats['rmse']['mean']:.6f} ± {aggregate_stats['rmse']['std']:.6f}")
    
    # Save results
    results_file = output_dir / "batch_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "aggregate_stats": aggregate_stats,
            "per_episode_metrics": all_metrics,
        }, f, indent=2)
    
    logging.info(f"\nResults saved to: {results_file}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

