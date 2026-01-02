"""DISCA Trainer - EM-Style Iterative Training with YOPO."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.feature_extractor import FeatureExtractor3D
from models.clustering import YOPOClustering
from utils.metrics import compute_all_metrics


class DISCATrainer:
    """Main trainer for DISCA with EM-style training and YOPO clustering."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        
        self.feature_extractor = FeatureExtractor3D(config).to(device)
        self.clustering = YOPOClustering(
            num_clusters=config["clustering"]["num_clusters"],
            feature_dim=config["model"]["feature_dim"],
            clustering_method=config["clustering"]["method"],
            config=config,
        ).to(device)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.setup_logging()
        
        print(f"DISCA Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Feature extractor params: {sum(p.numel() for p in self.feature_extractor.parameters()):,}")
        print(f"   Clusters: {config['clustering']['num_clusters']}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        optimizer_name = self.config["training"]["optimizer"]
        lr = float(self.config["training"]["learning_rate"])
        
        if optimizer_name == "adam":
            adam_config = self.config["training"]["adam"]
            return optim.Adam(
                self.feature_extractor.parameters(),
                lr=lr,
                betas=(float(adam_config["beta1"]), float(adam_config["beta2"])),
                weight_decay=float(adam_config["weight_decay"]),
            )
        elif optimizer_name == "sgd":
            sgd_config = self.config["training"]["sgd"]
            return optim.SGD(
                self.feature_extractor.parameters(),
                lr=lr,
                momentum=float(sgd_config["momentum"]),
                weight_decay=float(sgd_config["weight_decay"]),
                nesterov=sgd_config["nesterov"],
            )
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        if not self.config["training"]["scheduler"]["enabled"]:
            return None
        
        scheduler_type = self.config["training"]["scheduler"]["type"]
        
        if scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["training"]["scheduler"]["step_size"],
                gamma=self.config["training"]["scheduler"]["gamma"],
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["num_epochs"],
            )
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def setup_logging(self):
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.checkpoint_dir = self.output_dir / self.config["logging"]["checkpoint_dir"]
        self.visualization_dir = self.output_dir / self.config["evaluation"]["visualization"]["output_dir"]
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config["logging"]["tensorboard"]:
            tensorboard_dir = self.output_dir / self.config["logging"]["tensorboard_dir"]
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        else:
            self.writer = None
    
    def _warmup_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.feature_extractor.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Warmup {self.current_epoch}", leave=False)
        
        for batch_idx, (batch, _) in enumerate(pbar):
            batch = batch.to(self.device)
            features = self.feature_extractor(batch)
            
            feature_mean = features.mean(dim=0, keepdim=True)
            feature_var = ((features - feature_mean) ** 2).mean()
            variance_loss = -torch.log(feature_var + 1e-8)
            
            features_centered = features - feature_mean
            cov_matrix = torch.mm(features_centered.t(), features_centered) / (features.size(0) - 1)
            off_diag_mask = 1.0 - torch.eye(features.size(1), device=self.device)
            covariance_loss = (cov_matrix * off_diag_mask).pow(2).mean()
            
            pairwise_dist = torch.cdist(features, features)
            mask = 1.0 - torch.eye(features.size(0), device=self.device)
            uniformity_loss = -(pairwise_dist * mask).mean()
            
            loss = variance_loss + 0.01 * covariance_loss + 0.1 * uniformity_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            
            gradient_clip = self.config.get("training", {}).get("gradient_clip", 1.0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), gradient_clip)
            
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return {"warmup_loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    def extract_all_features(self, dataloader: DataLoader) -> Tuple[torch.Tensor, np.ndarray]:
        self.feature_extractor.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch, labels in tqdm(dataloader, desc="Extracting features", leave=False):
                batch = batch.to(self.device)
                features = self.feature_extractor(batch)
                
                if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
                    features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
                
                all_features.append(features.cpu())
                all_labels.extend(labels.numpy())
        
        return torch.cat(all_features, dim=0), np.array(all_labels)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        cluster_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.feature_extractor.train()
        total_loss = 0.0
        num_batches = 0
        
        if cluster_probs is not None:
            batch_size = train_loader.batch_size
            cluster_probs_batches = torch.split(cluster_probs, batch_size)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False)
        
        for batch_idx, (batch, _) in enumerate(pbar):
            batch = batch.to(self.device)
            features = self.feature_extractor(batch)
            
            if cluster_probs is not None and batch_idx < len(cluster_probs_batches):
                batch_cluster_probs = cluster_probs_batches[batch_idx].to(self.device)
            else:
                output = self.clustering(features, update_clusters=False)
                batch_cluster_probs = output["cluster_probs"]
            
            if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
                continue
            
            loss = self.clustering.compute_clustering_loss(features, batch_cluster_probs)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            
            gradient_clip = self.config.get("training", {}).get("gradient_clip", 1.0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), gradient_clip)
            
            has_nan_grad = any(
                torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad))
                for p in self.feature_extractor.parameters() if p.grad is not None
            )
            
            if has_nan_grad:
                self.optimizer.zero_grad()
                continue
            
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return {
            "train_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
    
    def evaluate(self, val_loader: DataLoader, compute_metrics: bool = True) -> Dict[str, float]:
        all_features, true_labels = self.extract_all_features(val_loader)
        all_features = all_features.to(self.device)
        
        output = self.clustering(all_features, update_clusters=True)
        pred_labels = output["cluster_labels"]
        val_loss = output["loss"].item()
        
        metrics = {"val_loss": val_loss}
        
        if compute_metrics:
            clustering_metrics = compute_all_metrics(
                pred_labels=pred_labels,
                true_labels=true_labels if (true_labels >= 0).any() else None,
                features=all_features.cpu().numpy(),
            )
            metrics.update(clustering_metrics)
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        num_epochs = self.config["training"]["num_epochs"]
        warmup_epochs = self.config["training"].get("warmup_epochs", 0)
        reassignment_freq = self.config["clustering"]["reassignment_frequency"]
        eval_freq = self.config["evaluation"]["eval_frequency"]
        save_freq = self.config["logging"]["save_frequency"]
        
        print(f"\nStarting DISCA training for {num_epochs} epochs")
        if warmup_epochs > 0:
            print(f"   Warmup period: {warmup_epochs} epochs (encoder pre-training)")
        print()
        
        cluster_probs = None
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            if epoch < warmup_epochs:
                print(f"\nWarmup epoch {epoch}/{warmup_epochs}: Training encoder (no clustering)")
                warmup_metrics = self._warmup_epoch(train_loader)
                epoch_time = time.time() - epoch_start_time
                print(f"   Warmup loss: {warmup_metrics['warmup_loss']:.4f} ({epoch_time:.1f}s)")
                if self.writer:
                    self.writer.add_scalar("warmup_loss", warmup_metrics['warmup_loss'], epoch)
                continue
            
            effective_epoch = epoch - warmup_epochs
            if effective_epoch % reassignment_freq == 0:
                print(f"\nE-step: Updating cluster assignments...")
                try:
                    all_features, _ = self.extract_all_features(train_loader)
                    all_features = all_features.to(self.device)
                    
                    if torch.any(torch.isnan(all_features)) or torch.any(torch.isinf(all_features)):
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.1
                        continue
                    
                    output = self.clustering(all_features, update_clusters=True)
                    cluster_probs = output["cluster_probs"]
                    
                    cluster_labels = output["cluster_labels"]
                    unique, counts = np.unique(cluster_labels, return_counts=True)
                    print(f"   Clusters updated: {len(unique)} active clusters")
                    print(f"   Distribution: {dict(zip(unique, counts))}")
                except ValueError:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    continue
            
            train_metrics = self.train_epoch(train_loader, cluster_probs)
            
            if self.scheduler:
                self.scheduler.step()
            
            if epoch % eval_freq == 0:
                val_metrics = self.evaluate(val_loader, compute_metrics=True)
                metrics = {**train_metrics, **val_metrics}
                
                if self.writer:
                    for key, value in metrics.items():
                        self.writer.add_scalar(key, value, epoch)
                
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"   Train Loss: {metrics['train_loss']:.4f}")
                print(f"   Val Loss:   {metrics['val_loss']:.4f}")
                
                if metrics["val_loss"] < self.best_loss:
                    self.best_loss = metrics["val_loss"]
                    self.epochs_without_improvement = 0
                    self.save_checkpoint("best_model.pth")
                    print(f"   New best model saved!")
                else:
                    self.epochs_without_improvement += eval_freq
            
            if epoch % save_freq == 0:
                self.save_checkpoint(f"model_epoch{epoch}.pth")
            
            early_stop_config = self.config["training"]["early_stopping"]
            if early_stop_config["enabled"]:
                if self.epochs_without_improvement >= early_stop_config["patience"]:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        print(f"\nTraining complete!")
        print(f"   Best validation loss: {self.best_loss:.4f}")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, filename: str):
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "feature_extractor_state_dict": self.feature_extractor.state_dict(),
            "cluster_centers": self.clustering.cluster_centers,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        self.clustering.cluster_centers = checkpoint["cluster_centers"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
