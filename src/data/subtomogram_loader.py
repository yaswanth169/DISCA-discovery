"""Subtomogram Dataset and DataLoader utilities."""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mrcfile


class SubtomogramDataset(Dataset):
    """Dataset for loading 3D subtomogram volumes."""
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        load_labels: bool = False,
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.load_labels = load_labels
        self.normalize = normalize
        
        self.file_paths = self._find_files()
        self.labels = self._load_labels() if load_labels else [-1] * len(self.file_paths)
        
        if len(self.file_paths) == 0:
            self._generate_synthetic_data()
        
        print(f"Loaded {len(self.file_paths)} subtomograms from {data_dir}")
    
    def _find_files(self) -> List[Path]:
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        files = sorted(self.data_dir.glob("*.mrc"))
        return files
    
    def _load_labels(self) -> List[int]:
        labels = []
        for fp in self.file_paths:
            label_file = fp.with_suffix(".txt")
            if label_file.exists():
                with open(label_file) as f:
                    labels.append(int(f.read().strip()))
            else:
                labels.append(-1)
        return labels
    
    def _generate_synthetic_data(self, num_samples: int = 1000, num_classes: int = 5):
        print(f"Generating {num_samples} synthetic subtomograms...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            class_id = i % num_classes
            volume = self._create_synthetic_volume(class_id)
            
            file_path = self.data_dir / f"subtomo_{i:05d}.mrc"
            with mrcfile.new(str(file_path), overwrite=True) as mrc:
                mrc.set_data(volume.astype(np.float32))
            
            label_file = file_path.with_suffix(".txt")
            with open(label_file, "w") as f:
                f.write(str(class_id))
        
        self.file_paths = self._find_files()
        self.labels = self._load_labels()
    
    def _create_synthetic_volume(self, class_id: int, size: int = 32) -> np.ndarray:
        volume = np.random.randn(size, size, size) * 0.1
        center = size // 2
        
        if class_id == 0:
            radius = size // 4
            z, y, x = np.ogrid[:size, :size, :size]
            mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
            volume[mask] += 1.0
        elif class_id == 1:
            half = size // 4
            volume[center-half:center+half, center-half:center+half, center-half:center+half] += 1.0
        elif class_id == 2:
            for axis in range(3):
                slices = [slice(center-1, center+1)] * 3
                slices[axis] = slice(size//4, 3*size//4)
                volume[tuple(slices)] += 0.8
        elif class_id == 3:
            radius = size // 3
            z, y, x = np.ogrid[:size, :size, :size]
            shell_outer = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
            shell_inner = (x - center)**2 + (y - center)**2 + (z - center)**2 <= (radius-2)**2
            volume[shell_outer & ~shell_inner] += 1.0
        else:
            for offset in [-size//6, size//6]:
                z, y, x = np.ogrid[:size, :size, :size]
                mask = (x - center)**2 + (y - center - offset)**2 + (z - center)**2 <= (size//6)**2
                volume[mask] += 0.8
        
        return volume
    
    def _load_mrc(self, path: Path) -> np.ndarray:
        try:
            with mrcfile.open(str(path), permissive=True) as mrc:
                return mrc.data.copy()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((32, 32, 32), dtype=np.float32)
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        mean = volume.mean()
        std = volume.std()
        if std > 1e-8:
            return (volume - mean) / std
        return volume - mean
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        volume = self._load_mrc(self.file_paths[idx])
        
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        if self.transform:
            volume = self.transform(volume)
        
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        label = self.labels[idx]
        
        return volume, label


class SubtomogramAugmentation:
    """Data augmentation for 3D volumes."""
    
    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        noise_std: float = 0.01,
    ):
        self.rotation = rotation
        self.flip = flip
        self.noise_std = noise_std
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if self.rotation:
            k = np.random.randint(0, 4)
            axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
            volume = np.rot90(volume, k=k, axes=axes)
        
        if self.flip:
            for axis in range(3):
                if np.random.random() > 0.5:
                    volume = np.flip(volume, axis=axis)
        
        if self.noise_std > 0:
            volume = volume + np.random.randn(*volume.shape) * self.noise_std
        
        return np.ascontiguousarray(volume)


def create_dataloaders(
    config: dict,
    data_dir: Optional[str] = None,
    load_labels: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    data_dir = data_dir or config["data"]["data_dir"]
    
    aug_config = config["data"].get("augmentation", {})
    transform = None
    if aug_config.get("enabled", False):
        transform = SubtomogramAugmentation(
            rotation=True,
            flip=True,
            noise_std=aug_config.get("gaussian_noise", 0.01),
        )
    
    dataset = SubtomogramDataset(
        data_dir=data_dir,
        transform=transform,
        load_labels=load_labels,
        normalize=config["data"].get("normalization", {}).get("method", "zscore") != "none",
    )
    
    train_split = config["data"].get("train_val_split", 0.8)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get("computing", {}).get("seed", 42))
    )
    
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config["data"].get("pin_memory", True),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["data"].get("pin_memory", True),
    )
    
    print(f"Created DataLoaders:")
    print(f"   Train: {train_size} samples, {len(train_loader)} batches")
    print(f"   Val:   {val_size} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader
