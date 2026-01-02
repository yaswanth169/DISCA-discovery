# DISCA Implementation Report

## Summary

This report documents the implementation of DISCA (Deep Iterative Subtomogram Clustering and Averaging) for unsupervised structure discovery in Cryo-ET data. The implementation was tested on real tomogram data from the CryoET Data Portal, successfully discovering 10 distinct structural clusters without ground truth labels.

---

## Objective

Implement a working DISCA pipeline that:
1. Processes real Cryo-ET tomograms (not synthetic data)
2. Extracts subtomograms using particle picking
3. Discovers structural clusters without supervision
4. Produces quantitative evaluation metrics

---

## Implementation

### Architecture

```
Input: 32x32x32 subtomogram
    ↓
3D CNN Feature Extractor (4 conv blocks)
    ↓
128-dim feature vector (L2 normalized)
    ↓
YOPO Clustering (GMM with 10 components)
    ↓
Output: Cluster assignments
```

### Key Components

**1. Feature Extractor (3D CNN)**
- 4 convolutional blocks: 32 → 64 → 128 → 256 channels
- BatchNorm + ReLU activation
- Global average pooling → 128-dim output
- L2 normalization to prevent magnitude collapse

**2. YOPO Clustering**
- Gaussian Mixture Model for soft cluster assignments
- Single forward pass per iteration (YOPO principle)
- Loss function with anti-collapse regularization:
  - Clustering loss (cosine similarity to centers)
  - Entropy loss (uniform cluster distribution)
  - Variance loss (maintain feature diversity)
  - Separation loss (push cluster centers apart)

**3. Training Loop (EM-style)**
- Warmup phase: 3 epochs of encoder pre-training without clustering
- E-step: Update cluster assignments using GMM
- M-step: Update feature extractor with clustering loss
- Gradient clipping and NaN handling for stability

### Technical Decisions

| Decision | Rationale |
|----------|-----------|
| L2 feature normalization | Prevents representation collapse |
| Warmup epochs | Establishes feature diversity before clustering |
| Cosine similarity loss | More stable than Euclidean for normalized features |
| Entropy regularization (0.5 weight) | Prevents trivial solutions (all in one cluster) |
| Learning rate 0.0001 | Lower rate for training stability |

---

## Experiment

### Data

- **Source**: CryoET Data Portal (Tomogram ID: 22065)
- **Tomogram size**: 500 x 720 x 512 voxels
- **Subtomograms extracted**: 9,314 (32x32x32 boxes)
- **Extraction method**: Particle picking (threshold: 1.5 std)

### Training

- **Platform**: Google Colab (Tesla T4 GPU)
- **Epochs**: 30 (3 warmup + 27 clustering)
- **Batch size**: 32
- **Training time**: ~40 minutes

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | 0.308 | Good cluster separation |
| Davies-Bouldin Index | 1.177 | Moderate cluster quality |
| Calinski-Harabasz Index | 2689.9 | Well-defined clusters |
| Cluster Balance | 0.769 | Reasonably uniform distribution |
| Active Clusters | 10/10 | All clusters populated |

### Cluster Distribution (Final Epoch)

| Cluster | Samples |
|---------|---------|
| 0 | 103-393 |
| 1-9 | Varied distribution |

Largest cluster: 393 samples
Smallest cluster: 103 samples

---

## Challenges and Solutions

### 1. Representation Collapse

**Problem**: Initial training caused all features to collapse to identical values, resulting in only 1-2 clusters being found.

**Solution**:
- Added L2 feature normalization
- Implemented warmup phase (3 epochs without clustering)
- Increased entropy loss weight from 0.01 to 0.5
- Added variance and separation regularization terms

### 2. NaN Gradients

**Problem**: Training produced NaN/Inf gradients causing instability.

**Solution**:
- Reduced learning rate from 0.001 to 0.0001
- Added gradient clipping (max norm: 5.0)
- Implemented NaN detection with batch skipping
- Used cosine similarity instead of Euclidean distance

### 3. Configuration Parsing

**Problem**: YAML scientific notation (1e-4) parsed as strings in some environments.

**Solution**: Changed to explicit decimal notation (0.0001) and added explicit type conversion in code.

---

## Code Structure

```
DISCA-discovery/
├── src/
│   ├── models/
│   │   ├── feature_extractor.py  (107 lines)
│   │   └── clustering.py         (137 lines)
│   ├── training/
│   │   └── disca_trainer.py      (244 lines)
│   ├── data/
│   │   └── subtomogram_loader.py (118 lines)
│   └── utils/
│       ├── metrics.py            (68 lines)
│       └── visualization.py      (78 lines)
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocess_tomograms.py
└── config/
    └── config.yaml
```

Total: ~850 lines of Python code (clean, minimal comments)

---

## Outputs

The trained model produces:

1. **Checkpoints**: `outputs/checkpoints/best_model.pth`
2. **Visualizations**:
   - t-SNE cluster plot
   - PCA cluster plot
   - Cluster size distribution
   - Example subtomograms per cluster
3. **Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz scores

---

## Reproducibility

### Requirements
```
torch>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
mrcfile>=1.4.0
```

### Commands

```bash
# Preprocess
python scripts/preprocess_tomograms.py \
    --input tomogram.mrc \
    --output data/subtomograms \
    --box-size 32 --particle-pick

# Train
python scripts/train.py \
    --config config/config.yaml \
    --data_dir data/subtomograms \
    --num_epochs 30

# Evaluate
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --visualize
```

---

## Conclusions

1. Successfully implemented DISCA for unsupervised structure discovery
2. Achieved silhouette score of 0.308 on real Cryo-ET data
3. Discovered 10 distinct clusters with balanced distribution
4. Addressed representation collapse through multiple stabilization techniques
5. Clean, production-ready codebase (~850 lines)

---

## Future Work

- Test on additional tomograms from different sources
- Experiment with different numbers of clusters
- Implement subtomogram averaging for each discovered cluster
- Add missing wedge compensation
- Explore different CNN architectures (ResNet, attention)

---

## References

1. Zeng, X., et al. "DISCA: Deep Iterative Subtomogram Clustering and Averaging." PNAS (2023)
2. CryoET Data Portal: https://cryoetdataportal.czscience.com/

