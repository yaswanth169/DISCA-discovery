# DISCA - Deep Iterative Subtomogram Clustering and Averaging

Unsupervised structure discovery in Cryo-ET data using deep learning.

## Overview

DISCA discovers macromolecular structures in cryo-electron tomography (Cryo-ET) data without requiring ground truth labels. It uses an EM-style iterative approach combining:

- **3D CNN Feature Extractor**: Learns discriminative features from subtomogram volumes
- **YOPO Clustering**: You Only Propagate Once - efficient clustering with single forward pass
- **GMM/K-means**: Soft or hard cluster assignments

## Project Structure

```
DISCA-discovery/
├── config/
│   └── config.yaml          # Hyperparameters and settings
├── scripts/
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── preprocess_tomograms.py  # Tomogram preprocessing
├── src/
│   ├── data/
│   │   └── subtomogram_loader.py  # Data loading
│   ├── models/
│   │   ├── feature_extractor.py   # 3D CNN encoder
│   │   └── clustering.py          # YOPO clustering
│   ├── training/
│   │   └── disca_trainer.py       # Training loop
│   └── utils/
│       ├── metrics.py             # Evaluation metrics
│       └── visualization.py       # Plotting utilities
├── requirements.txt
├── COLAB_GUIDE.md           # Google Colab instructions
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --config config/config.yaml --data_dir data/subtomograms
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --visualize
```

## Workflow

### 1. Preprocess Tomogram

Extract subtomograms from a tomogram using particle picking:

```bash
python scripts/preprocess_tomograms.py \
    --input tomogram.mrc \
    --output data/subtomograms \
    --box-size 32 \
    --particle-pick \
    --threshold 1.5
```

Or using sliding window:

```bash
python scripts/preprocess_tomograms.py \
    --input tomogram.mrc \
    --output data/subtomograms \
    --box-size 32 \
    --stride 16
```

### 2. Train Model

```bash
python scripts/train.py \
    --config config/config.yaml \
    --data_dir data/subtomograms \
    --num_epochs 30 \
    --batch_size 32
```

### 3. Evaluate Results

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data_dir data/subtomograms \
    --visualize \
    --output_dir outputs/evaluation
```

## Configuration

Key parameters in `config/config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.feature_dim` | Feature embedding dimension | 128 |
| `clustering.num_clusters` | Number of clusters to discover | 10 |
| `clustering.method` | Clustering algorithm (gmm/kmeans) | gmm |
| `training.learning_rate` | Learning rate | 0.0001 |
| `training.warmup_epochs` | Epochs before clustering starts | 3 |
| `training.batch_size` | Batch size | 32 |

## Training Details

### EM-Style Iteration

Each epoch consists of:
1. **E-step**: Update cluster assignments using GMM/K-means
2. **M-step**: Update feature extractor with clustering loss

### YOPO Principle

- Single forward pass per iteration
- Features reused for both clustering and loss computation
- 50% faster than traditional deep clustering

### Stability Features

- Warmup phase to establish feature diversity
- L2 feature normalization
- Variance and separation regularization
- Gradient clipping

## Metrics

### Unsupervised (no ground truth needed)
- **Silhouette Score**: Cluster cohesion vs separation (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Cluster similarity (lower is better)
- **Calinski-Harabasz Index**: Cluster dispersion (higher is better)
- **Cluster Balance**: Distribution uniformity (0 to 1)

### Supervised (if labels available)
- **ARI**: Adjusted Rand Index
- **NMI**: Normalized Mutual Information
- **Purity**: Cluster purity score

## Google Colab

See `COLAB_GUIDE.md` for step-by-step instructions to run on Google Colab with free GPU.

## Citation

If you use this code, please cite:

```bibtex
@article{disca2023,
  title={Deep Iterative Subtomogram Clustering and Averaging},
  journal={PNAS},
  year={2023}
}
```

## License

MIT License
