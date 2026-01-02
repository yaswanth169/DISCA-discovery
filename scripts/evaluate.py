#!/usr/bin/env python
"""Evaluate trained DISCA model."""

import argparse
import sys
import os
from pathlib import Path
from typing import Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import torch
from tqdm import tqdm

from models import FeatureExtractor3D, YOPOClustering
from data import create_dataloaders
from utils import compute_all_metrics, visualize_clusters, plot_cluster_statistics, save_cluster_examples


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DISCA model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    feature_extractor = FeatureExtractor3D(config).to(device)
    feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
    feature_extractor.eval()
    
    clustering = YOPOClustering(
        num_clusters=config["clustering"]["num_clusters"],
        feature_dim=config["model"]["feature_dim"],
        clustering_method=config["clustering"]["method"],
        config=config,
    ).to(device)
    clustering.cluster_centers = checkpoint["cluster_centers"]
    
    print(f"   Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Best loss: {checkpoint['best_loss']:.4f}")
    
    return feature_extractor, clustering, config


def extract_features_and_cluster(
    feature_extractor,
    clustering,
    dataloader,
    device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\nExtracting features and clustering...")
    
    all_features = []
    all_labels = []
    all_volumes = []
    
    with torch.no_grad():
        for batch, labels in tqdm(dataloader, desc="Processing"):
            batch = batch.to(device)
            features = feature_extractor(batch)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_volumes.append(batch.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    true_labels = np.array(all_labels)
    volumes = np.concatenate(all_volumes, axis=0)
    
    features_tensor = torch.from_numpy(features).to(device)
    output = clustering(features_tensor, update_clusters=True)
    pred_labels = output["cluster_labels"]
    
    print(f"   Processed {len(features)} samples")
    print(f"   Found {len(np.unique(pred_labels))} clusters")
    
    return features, pred_labels, true_labels, volumes


def print_metrics(metrics: dict):
    print("\n" + "=" * 60)
    print("CLUSTERING METRICS")
    print("=" * 60)
    
    supervised = ["ari", "nmi", "purity", "accuracy"]
    unsupervised = ["silhouette", "davies_bouldin", "calinski_harabasz", "cluster_balance"]
    stats = ["num_clusters", "largest_cluster_size", "smallest_cluster_size"]
    
    has_supervised = any(m in metrics for m in supervised)
    if has_supervised:
        print("\nSupervised Metrics:")
        for m in supervised:
            if m in metrics:
                print(f"   {m:20s}: {metrics[m]:.4f}")
    
    print("\nUnsupervised Metrics:")
    for m in unsupervised:
        if m in metrics:
            extra = " (lower is better)" if m == "davies_bouldin" else ""
            print(f"   {m:20s}: {metrics[m]:.4f}{extra}")
    
    print("\nCluster Statistics:")
    for m in stats:
        if m in metrics:
            print(f"   {m:20s}: {int(metrics[m])}")
    
    print("=" * 60)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("DISCA Model Evaluation")
    print("=" * 70)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    feature_extractor, clustering, config = load_checkpoint(args.checkpoint, device)
    
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    
    print(f"\nLoading data from: {config['data']['data_dir']}")
    train_loader, val_loader = create_dataloaders(config, load_labels=False)
    
    features, pred_labels, true_labels, volumes = extract_features_and_cluster(
        feature_extractor, clustering, val_loader, device
    )
    
    print("\nComputing metrics...")
    metrics = compute_all_metrics(
        pred_labels=pred_labels,
        true_labels=true_labels if (true_labels >= 0).any() else None,
        features=features,
    )
    
    print_metrics(metrics)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"\nSaved metrics to: {metrics_file}")
    
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        print("   1. Creating t-SNE visualization...")
        visualize_clusters(features, pred_labels, str(vis_dir), method="tsne")
        
        print("   2. Creating PCA visualization...")
        visualize_clusters(features, pred_labels, str(vis_dir), method="pca")
        
        print("   3. Creating cluster statistics plot...")
        plot_cluster_statistics(pred_labels, str(vis_dir))
        
        print("   4. Saving example subtomograms...")
        examples_dir = vis_dir / "cluster_examples"
        save_cluster_examples(volumes, pred_labels, str(examples_dir), num_examples=5)
        print(f"Saved cluster examples to {examples_dir}")
        
        print(f"\n   Visualizations saved to: {vis_dir}")
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"   Metrics:         metrics.txt")
    if args.visualize:
        print(f"   Visualizations:  visualizations/")
    
    print(f"\nKey Results:")
    if "silhouette" in metrics:
        print(f"   Silhouette Score:     {metrics['silhouette']:.3f}")
    print(f"   Number of Clusters:   {int(metrics['num_clusters'])}")
    print(f"   Cluster Balance:      {metrics['cluster_balance']:.3f}")


if __name__ == "__main__":
    main()
