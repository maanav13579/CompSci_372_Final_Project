# Setup Guide

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~15GB disk space for Food-101 dataset

## Environment Setup

### Option 1: pip (recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda

```bash
conda create -n food-ai python=3.10
conda activate food-ai
pip install -r requirements.txt
```

## Dataset Download

### Food-101 (Required)

1. Download from official source:
```bash
wget https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
```

2. Extract to data directory:
```bash
tar -xzf food-101.tar.gz -C data/
mv data/food-101 data/food101
```

3. Verify structure:
```
data/food101/
├── images/
│   ├── apple_pie/
│   │   ├── 134.jpg
│   │   └── ...
│   └── ... (101 folders)
└── meta/
    ├── classes.txt
    ├── train.txt
    └── test.txt
```

### Nutrition5k (Optional)

For better calorie regression, download Nutrition5k:

1. Request access at: https://github.com/google-research-datasets/Nutrition5k
2. Download and extract to `data/nutrition5k/`

If unavailable, the system falls back to class-average calorie lookup.

## Verify Installation

```bash
cd src
python -c "
import torch
from models import create_classifier, create_regressor

# Check CUDA
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test models
clf = create_classifier()
reg = create_regressor()

x = torch.randn(1, 3, 224, 224)
print(f'Classifier output: {clf(x).shape}')
print(f'Regressor output: {reg(x).shape}')
print('Setup complete!')
"
```

## Common Issues

### Out of Memory
- Reduce batch size: `--batch_size 16`
- Use gradient checkpointing
- Enable mixed precision: `--use_amp`

### Slow Training
- Increase `--num_workers` (try 8)
- Use SSD storage for dataset
- Enable mixed precision training

### Dataset Not Found
- Check path in `--data_root`
- Verify Food-101 extraction completed
- Check file permissions

## Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1080 (8GB) | RTX 3080 (10GB)+ |
| RAM | 16GB | 32GB |
| Storage | HDD | NVMe SSD |
| CPU | 4 cores | 8+ cores |

## Training Time Estimates

With RTX 3080:
- Classifier (30 epochs): ~2-3 hours
- Regressor (25 epochs): ~1-2 hours

With CPU only:
- Classifier: ~24+ hours
- Not recommended for training
