# DISCA - Google Colab Guide

Step-by-step commands for running DISCA on Google Colab.

## Step 1: Setup

```python
import torch
import os

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

!pip install -q mrcfile scikit-learn pyyaml tqdm matplotlib seaborn scipy cryoet-data-portal
```

## Step 2: Upload Project

Upload the DISCA-discovery folder to Colab, or clone from repository:

```python
# Navigate to project
%cd /content/DISCA-discovery
```

## Step 3: Download Tomogram

```python
from cryoet_data_portal import Client, Tomogram
import os

os.makedirs('data/real', exist_ok=True)

client = Client()
tomogram = Tomogram.get_by_id(client, 22065)
tomogram.download_mrcfile(dest_path='data/real/tomogram_003.mrc')
print("Download complete")
```

## Step 4: Verify File

```python
import mrcfile

with mrcfile.open('data/real/tomogram_003.mrc', permissive=True) as mrc:
    print(f"Shape: {mrc.data.shape}")
    print(f"Range: [{mrc.data.min():.2f}, {mrc.data.max():.2f}]")
```

## Step 5: Extract Subtomograms

```python
!python scripts/preprocess_tomograms.py \
    --input data/real/tomogram_003.mrc \
    --output data/real_subtomograms \
    --box-size 32 \
    --particle-pick \
    --threshold 1.5
```

## Step 6: Train Model

```python
!python scripts/train.py \
    --config config/config.yaml \
    --data_dir data/real_subtomograms \
    --num_epochs 30 \
    --batch_size 32
```

Training time: ~30-40 minutes on Tesla T4

## Step 7: Evaluate

```python
!python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data_dir data/real_subtomograms \
    --visualize \
    --output_dir outputs/evaluation
```

## Step 8: Download Results

```python
import shutil
from google.colab import files

shutil.make_archive('disca_results', 'zip', 'outputs')
files.download('disca_results.zip')
```

## Troubleshooting

### Out of memory
Reduce batch size:
```python
!python scripts/train.py --batch_size 16 ...
```

### Too slow
Reduce epochs:
```python
!python scripts/train.py --num_epochs 15 ...
```

### Not enough subtomograms
Lower threshold or use sliding window:
```python
!python scripts/preprocess_tomograms.py --threshold 1.0 ...
# or
!python scripts/preprocess_tomograms.py --stride 16 ...
```

## Expected Results

- Silhouette Score: 0.2-0.4
- Number of Clusters: 10
- Cluster Balance: 0.6-0.9
- Training Loss: Negative (expected due to cosine similarity loss)
