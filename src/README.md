# 🍽️ AI Food Recognizer & Calorie Estimator

A dual-network machine learning system that classifies food images and estimates their calorie content.

## Overview

This project implements:
1. **Food Classification Network**: ResNet50/ViT trained on Food-101 dataset
2. **Calorie Regression Network**: Estimates calories from food images
3. **Unified Pipeline**: Combines both networks with multiple integration strategies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Food-101 Dataset

```bash
# Download from https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
wget https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz -C data/
mv data/food-101 data/food101
```

### 3. Train Classifier

```bash
cd src
python train_classifier.py \
    --data_root ../data/food101 \
    --backbone resnet50 \
    --epochs 30 \
    --batch_size 32 \
    --lr 3e-4
```

### 4. Train Regressor

Using Food-101 with calorie lookup:
```bash
python train_regressor.py \
    --data_root ../data/food101 \
    --dataset_type food101_lookup \
    --calorie_map ../data/calorie_map.csv \
    --epochs 25
```

Or with Nutrition5k (if available):
```bash
python train_regressor.py \
    --data_root ../data/nutrition5k \
    --dataset_type nutrition5k \
    --epochs 25
```

### 5. Run Inference

```python
from pipeline import FoodCaloriePipeline

pipeline = FoodCaloriePipeline(
    classifier_path="artifacts/models/classifier_best.pth",
    regressor_path="artifacts/models/regressor_best.pth",
    calorie_map_path="data/calorie_map.csv"
)

result = pipeline.predict("path/to/food/image.jpg", strategy="ensemble")
print(f"Food: {result['class_name']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Calories: {result['calories']:.0f} kcal")
```

## Project Structure

```
ai-food-recognizer/
├── data/                          # Datasets
│   ├── food101/                   # Food-101 dataset
│   ├── nutrition5k/               # Nutrition5k dataset (optional)
│   └── calorie_map.csv            # Class → calorie mapping
├── src/
│   ├── data/                      # Data loading utilities
│   │   ├── transforms.py          # Image augmentations
│   │   ├── classifier_dset.py     # Food-101 dataset
│   │   └── regressor_dset.py      # Calorie dataset
│   ├── models/
│   │   ├── classifier.py          # Classification model
│   │   └── regressor.py           # Regression model
│   ├── pipeline/
│   │   └── predict.py             # Unified inference pipeline
│   ├── train_classifier.py        # Classification training
│   └── train_regressor.py         # Regression training
├── notebooks/                     # Jupyter notebooks for analysis
├── artifacts/                     # Checkpoints, logs, figures
└── requirements.txt
```

## Model Architectures

### Classifier
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Head**: Dropout(0.3) → Linear(2048, 101)
- **Loss**: Cross-Entropy with label smoothing (0.1)
- **Target**: ≥85% Top-1 accuracy on Food-101

### Regressor
- **Backbone**: Frozen ResNet50 feature extractor
- **Head**: Linear(2048, 256) → ReLU → Dropout → Linear(256, 1)
- **Loss**: Huber loss (robust to outliers)
- **Target**: Lower RMSE than class-average lookup baseline

## Prediction Strategies

| Strategy | Description |
|----------|-------------|
| `classifier_only` | Classify → lookup calories from table |
| `regressor_only` | Direct calorie prediction from image |
| `ensemble` | Weighted combination of lookup + regression |
| `confidence_switch` | Use lookup if confident, else regressor |

## Training Tips

- Use mixed precision (`--use_amp`) for faster training
- Start with frozen backbone for regressor, then fine-tune
- Monitor TensorBoard for loss curves: `tensorboard --logdir artifacts/`
- Augmentation is crucial for generalization

## Evaluation

After training, check:
- `artifacts/class_logs/*/results.json` - Classification metrics
- `artifacts/reg_logs/*/results.json` - Regression metrics
- Run evaluation notebooks in `notebooks/` for detailed analysis

## Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| Classifier Top-1 | ~85% | Food-101 test set |
| Classifier Top-5 | ~96% | Food-101 test set |
| Regressor MAE | <80 kcal | Depends on dataset |
| Regressor MAPE | <20% | Relative error |

## License

MIT License - See LICENSE file.

## Acknowledgments

- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Nutrition5k Dataset](https://github.com/google-research-datasets/Nutrition5k)
