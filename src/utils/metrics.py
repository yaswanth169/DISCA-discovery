"""Clustering evaluation metrics."""

from typing import Dict, Optional
from collections import Counter
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_all_metrics(
    pred_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all clustering metrics."""
    metrics = {}
    
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        metrics["ari"] = adjusted_rand_score(true_labels, pred_labels)
        metrics["nmi"] = normalized_mutual_info_score(true_labels, pred_labels)
        metrics["purity"] = compute_purity(pred_labels, true_labels)
        metrics["accuracy"] = compute_cluster_accuracy(pred_labels, true_labels)
    
    if features is not None and len(np.unique(pred_labels)) > 1:
        try:
            metrics["silhouette"] = silhouette_score(features, pred_labels)
            metrics["davies_bouldin"] = davies_bouldin_score(features, pred_labels)
            metrics["calinski_harabasz"] = calinski_harabasz_score(features, pred_labels)
        except Exception:
            pass
    
    metrics["cluster_balance"] = compute_cluster_balance(pred_labels)
    
    cluster_counts = Counter(pred_labels)
    metrics["num_clusters"] = len(cluster_counts)
    metrics["largest_cluster_size"] = max(cluster_counts.values())
    metrics["smallest_cluster_size"] = min(cluster_counts.values())
    
    return metrics


def compute_purity(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Compute clustering purity."""
    total = len(pred_labels)
    purity = 0.0
    
    for cluster_id in np.unique(pred_labels):
        cluster_true = true_labels[pred_labels == cluster_id]
        if len(cluster_true) > 0:
            purity += Counter(cluster_true).most_common(1)[0][1]
    
    return purity / total


def compute_cluster_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Compute clustering accuracy with Hungarian matching."""
    from scipy.optimize import linear_sum_assignment
    
    n_clusters = max(pred_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = -np.sum((pred_labels == i) & (true_labels == j))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    correct = 0
    for i, j in zip(row_ind, col_ind):
        correct += np.sum((pred_labels == i) & (true_labels == j))
    
    return correct / len(pred_labels)


def compute_cluster_balance(cluster_labels: np.ndarray) -> float:
    """Compute cluster balance (1.0 = perfectly uniform)."""
    cluster_counts = Counter(cluster_labels)
    K = len(cluster_counts)
    N = len(cluster_labels)
    
    if K == 0:
        return 0.0
    
    ideal = 1.0 / K
    variance = sum(((c / N) - ideal) ** 2 for c in cluster_counts.values())
    return 1.0 / (1.0 + K * variance)
