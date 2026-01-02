#!/usr/bin/env python
"""Train DISCA model."""

import argparse
import sys
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import yaml

from training import DISCATrainer
from data import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train DISCA model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_clusters", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("DISCA: Deep Iterative Subtomogram Clustering and Averaging")
    print("=" * 70)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"\nLoading configuration from: {args.config}")
    
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
        print(f"   Overriding data_dir: {args.data_dir}")
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
        print(f"   Overriding num_epochs: {args.num_epochs}")
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
        print(f"   Overriding batch_size: {args.batch_size}")
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
        print(f"   Overriding learning_rate: {args.learning_rate}")
    if args.num_clusters:
        config["clustering"]["num_clusters"] = args.num_clusters
        print(f"   Overriding num_clusters: {args.num_clusters}")
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    seed = config.get("computing", {}).get("seed", 42)
    print(f"\nSetting random seed: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"\nData loaded:")
    print(f"   Training samples:   {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Batch size:         {config['training']['batch_size']}")
    
    print("\nBuilding DISCA trainer...")
    trainer = DISCATrainer(config, device)
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print(f"\nTraining Configuration:")
    print(f"   Epochs:              {config['training']['num_epochs']}")
    print(f"   Learning rate:       {config['training']['learning_rate']}")
    print(f"   Optimizer:           {config['training']['optimizer']}")
    print(f"   Number of clusters:  {config['clustering']['num_clusters']}")
    print(f"   Clustering method:   {config['clustering']['method']}")
    print(f"   Feature dimension:   {config['model']['feature_dim']}")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    print(f"\nOutputs saved to: {config['experiment']['output_dir']}")
    print(f"   Checkpoints:     {config['logging']['checkpoint_dir']}")
    print(f"   TensorBoard:     {config['logging']['tensorboard_dir']}")
    print(f"   Visualizations:  {config['evaluation']['visualization']['output_dir']}")
    
    print("\nNext steps:")
    print("   1. View training curves:  tensorboard --logdir outputs/runs")
    print("   2. Evaluate model:        python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print("   3. Visualize clusters:    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --visualize")


if __name__ == "__main__":
    main()
