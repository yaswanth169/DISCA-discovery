"""YOPO Clustering - You Only Propagate Once."""

from typing import Optional, Tuple, Dict
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class YOPOClustering(nn.Module):
    """YOPO-based clustering module for deep iterative clustering."""
    
    def __init__(
        self,
        num_clusters: int,
        feature_dim: int,
        clustering_method: str = "gmm",
        config: Optional[dict] = None,
    ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.clustering_method = clustering_method
        self.config = config or {}
        
        self.register_buffer("cluster_centers", torch.randn(num_clusters, feature_dim))
        
        if clustering_method == "gmm":
            gmm_config = self.config.get("clustering", {}).get("gmm", {})
            self.clusterer = GaussianMixture(
                n_components=num_clusters,
                covariance_type=gmm_config.get("covariance_type", "diag"),
                max_iter=int(gmm_config.get("max_iter", 100)),
                n_init=int(gmm_config.get("n_init", 10)),
                reg_covar=float(gmm_config.get("reg_covar", 1e-6)),
            )
        elif clustering_method == "kmeans":
            kmeans_config = self.config.get("clustering", {}).get("kmeans", {})
            self.clusterer = KMeans(
                n_clusters=num_clusters,
                max_iter=int(kmeans_config.get("max_iter", 300)),
                n_init=int(kmeans_config.get("n_init", 10)),
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        loss_weights = self.config.get("yopo", {}).get("loss_weights", {})
        self.w_clustering = loss_weights.get("clustering_loss", 1.0)
        self.w_consistency = loss_weights.get("consistency_loss", 0.05)
        self.w_entropy = loss_weights.get("entropy_loss", 0.5)
    
    def update_clusters(self, features: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        features_np = features.detach().cpu().numpy()
        
        if np.any(np.isnan(features_np)) or np.any(np.isinf(features_np)):
            features_np = np.nan_to_num(features_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.clusterer.fit(features_np)
        cluster_labels = self.clusterer.predict(features_np)
        
        if self.clustering_method == "gmm":
            cluster_probs = self.clusterer.predict_proba(features_np)
        else:
            distances = self._compute_distances_to_centers(features)
            cluster_probs = F.softmax(-distances / 0.1, dim=1).detach().cpu().numpy()
        
        if hasattr(self.clusterer, 'cluster_centers_'):
            centers = self.clusterer.cluster_centers_
        else:
            centers = self.clusterer.means_
        
        self.cluster_centers = torch.from_numpy(centers).float().to(features.device)
        cluster_probs = torch.from_numpy(cluster_probs).float().to(features.device)
        
        return cluster_probs, cluster_labels
    
    def _compute_distances_to_centers(self, features: torch.Tensor) -> torch.Tensor:
        features_expanded = features.unsqueeze(1)
        centers_expanded = self.cluster_centers.unsqueeze(0)
        distances = torch.sqrt(torch.sum((features_expanded - centers_expanded) ** 2, dim=2))
        return distances
    
    def compute_clustering_loss(
        self,
        features: torch.Tensor,
        cluster_probs: torch.Tensor,
    ) -> torch.Tensor:
        cluster_centers_normalized = F.normalize(self.cluster_centers, p=2, dim=1)
        
        similarity = torch.mm(features, cluster_centers_normalized.t())
        distances = 1.0 - similarity
        clustering_loss = torch.sum(cluster_probs * distances) / features.size(0)
        
        consistency_loss = -torch.mean(
            torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1)
        )
        
        cluster_counts = torch.sum(cluster_probs, dim=0)
        cluster_dist = cluster_counts / (torch.sum(cluster_counts) + 1e-8)
        entropy_loss = torch.sum(cluster_dist * torch.log(cluster_dist + 1e-8))
        
        feature_mean = features.mean(dim=0, keepdim=True)
        feature_var = ((features - feature_mean) ** 2).mean()
        variance_loss = -torch.log(feature_var + 1e-8)
        
        center_distances = torch.cdist(
            cluster_centers_normalized.unsqueeze(0),
            cluster_centers_normalized.unsqueeze(0)
        ).squeeze(0)
        mask = 1.0 - torch.eye(self.num_clusters, device=features.device)
        separation_loss = -torch.sum(center_distances * mask) / (mask.sum() + 1e-8)
        
        total_loss = (
            self.w_clustering * clustering_loss +
            self.w_consistency * consistency_loss +
            self.w_entropy * entropy_loss +
            0.1 * variance_loss +
            0.05 * separation_loss
        )
        
        return total_loss
    
    def forward(
        self,
        features: torch.Tensor,
        update_clusters: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if update_clusters:
            cluster_probs, cluster_labels = self.update_clusters(features)
        else:
            distances = self._compute_distances_to_centers(features)
            cluster_probs = F.softmax(-distances / 0.1, dim=1)
            cluster_labels = torch.argmax(cluster_probs, dim=1).cpu().numpy()
        
        loss = self.compute_clustering_loss(features, cluster_probs)
        
        return {
            "cluster_probs": cluster_probs,
            "cluster_labels": cluster_labels,
            "loss": loss,
            "distances": self._compute_distances_to_centers(features),
        }


def compute_cluster_purity(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    total_samples = len(cluster_labels)
    purity = 0.0
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        if len(cluster_true_labels) > 0:
            most_common_count = Counter(cluster_true_labels).most_common(1)[0][1]
            purity += most_common_count
    
    return purity / total_samples


def compute_cluster_balance(cluster_labels: np.ndarray) -> float:
    cluster_counts = Counter(cluster_labels)
    K = len(cluster_counts)
    N = len(cluster_labels)
    
    if K == 0:
        return 0.0
    
    ideal_size = 1.0 / K
    variance = sum(((count / N) - ideal_size) ** 2 for count in cluster_counts.values())
    return 1.0 / (1.0 + K * variance)
