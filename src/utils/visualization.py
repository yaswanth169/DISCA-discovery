"""Visualization utilities for DISCA."""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    method: str = "tsne",
):
    """Create 2D visualization of clusters."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    else:
        reducer = PCA(n_components=2)
    
    coords = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(f'Cluster Visualization ({method.upper()})')
    plt.tight_layout()
    plt.savefig(output_dir / f'{method}.png', dpi=150)
    plt.close()


def plot_cluster_statistics(labels: np.ndarray, output_dir: str):
    """Plot cluster size distribution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.xticks(unique)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_stats.png', dpi=150)
    plt.close()


def save_cluster_examples(
    volumes: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    num_examples: int = 5,
):
    """Save example slices from each cluster."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_labels = np.unique(labels)
    
    for cluster_id in unique_labels:
        cluster_dir = output_dir / f'cluster_{cluster_id}'
        cluster_dir.mkdir(exist_ok=True)
        
        cluster_indices = np.where(labels == cluster_id)[0]
        sample_indices = cluster_indices[:num_examples]
        
        for i, idx in enumerate(sample_indices):
            volume = volumes[idx]
            if volume.ndim == 4:
                volume = volume[0]
            
            mid_slice = volume[volume.shape[0] // 2]
            
            plt.figure(figsize=(4, 4))
            plt.imshow(mid_slice, cmap='gray')
            plt.axis('off')
            plt.savefig(cluster_dir / f'example_{i}.png', dpi=100, bbox_inches='tight')
            plt.close()


def visualize_training_history(history: dict, output_dir: str):
    """Plot training curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curves')
    
    if 'silhouette' in history:
        axes[1].plot(history['silhouette'], label='Silhouette')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].set_title('Clustering Metrics')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()
