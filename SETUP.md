# Setup Guide

This guide provides instructions for setting up and running the AI Food Recognizer & Calorie Estimator project.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training; CPU works for inference)
- Google account (for Colab) or local machine with ~20GB disk space

## Option 1: Google Colab (Recommended)

### Step 1: Open in Colab

Upload the project folder to Google Drive or clone from the repository.

### Step 2: Mount Drive and Setup Environment

```python
from google.colab import drive
drive.mount('/content/drive')

import sys, os
PROJECT_ROOT = '{path to project folder}'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
os.chdir(PROJECT_ROOT)
```

### Step 3: Install Dependencies

```python
!pip install -q torch torchvision timm grad-cam tensorboard tqdm gradio
```

### Step 4: Download Food-101 Dataset (for training only)

```python
!wget -q https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar -xzf food-101.tar.gz -C /content/
!ln -sf /content/food-101 data/food101
!rm food-101.tar.gz
```

### Step 5: Verify Setup

```python
import torch
from models import create_classifier, create_regressor

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test model creation
clf = create_classifier()
reg = create_regressor()
print("Models created successfully!")
```

## Option 2: Local Installation

### Step 1: Clone Repository

```bash
git clone [repository-url]
cd CompSci_372_Final_Project
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Datasets

**Food-101 (for classifier training):**
```bash
wget https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz -C data/
mv data/food-101 data/food101
rm food-101.tar.gz
```

**Nutrition5k (for regressor training):**
- Already included in data/ directory

## Running the Project

### Training (Pre-trained models provided)

**Train Classifier:**
```bash
PYTHONPATH=src python src/train_classifier.py \
    --data_root data/food101 \
    --epochs 30 \
    --batch_size 32
```

**Train Regressor:**
```bash
PYTHONPATH=src python src/train_regressor.py \
    --data_root data/nutrition5k \
    --dataset_type nutrition5k \
    --epochs 25
```

### Inference

**Using Pre-trained Models:**
```python
from pipeline import FoodCaloriePipeline

pipeline = FoodCaloriePipeline(
    classifier_path="artifacts/models/classifier_best.pth",
    regressor_path="artifacts/models/regressor_best.pth",
)

result = pipeline.predict("path/to/food/image.jpg")
```

### Running Baselines

**Classifier Baseline:**
```bash
PYTHONPATH=src python src/baseline_classifier.py --data_root data/food101
```

**Regressor Baseline:**
```bash
PYTHONPATH=src python src/baseline_regressor.py \
    --data_root data/nutrition5k \
    --model_path artifacts/models/regressor_best.pth
```

### Web Demo

```python
!pip install gradio
import gradio as gr
from pipeline import FoodCaloriePipeline

pipeline = FoodCaloriePipeline(
    classifier_path="artifacts/models/classifier_best.pth",
    regressor_path="artifacts/models/regressor_best.pth",
)

def predict(image):
    result = pipeline.predict(image)
    return f"Food: {result['class_name']}\nConfidence: {result['confidence']:.1%}\nCalories: {result['calories']:.0f} kcal"

gr.Interface(fn=predict, inputs=gr.Image(type="filepath"), outputs="text").launch(share=True)
```

## Project Structure

```
ai-food-recognizer/
├── src/
│   ├── data/                    # Data loading utilities
│   │   ├── classifier_dset.py   # Food-101 dataset
│   │   ├── regressor_dset.py    # Nutrition5k dataset
│   │   └── transforms.py        # Image augmentations
│   ├── models/                  # Model architectures
│   │   ├── classifier.py        # ResNet50 classifier
│   │   └── regressor.py         # Calorie regressor
│   ├── pipeline/                # Inference pipeline
│   │   └── predict.py           # Combined prediction
│   ├── train_classifier.py      # Classifier training
│   ├── train_regressor.py       # Regressor training
│   ├── baseline_classifier.py   # Classifier baseline
│   └── baseline_regressor.py    # Regressor baseline
├── data/                        # Datasets (not included)
├── artifacts/                   # Trained models and figures
│   ├── models/
│   └── figures/
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
└── requirements.txt
```
