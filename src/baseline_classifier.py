"""
Baseline classifier using Logistic Regression on frozen ResNet50 features.

This baseline extracts features from a pretrained ResNet50 (no fine-tuning)
and trains a simple logistic regression classifier on top.

Usage:
    python baseline_classifier.py --data_root data/food101
"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data import get_food101_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Classifier")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for logistic regression")
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    return parser.parse_args()


class FeatureExtractor(nn.Module):
    """Frozen ResNet50 feature extractor."""
    
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten to (batch, 2048)


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract features from all images in dataloader."""
    model.eval()
    all_features = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_food101_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Initialize feature extractor
    print("Initializing frozen ResNet50 feature extractor...")
    feature_extractor = FeatureExtractor().to(device)
    
    # Extract features
    print("\nExtracting training features...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    print(f"Train features shape: {train_features.shape}")
    
    print("\nExtracting validation features...")
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)
    
    print("\nExtracting test features...")
    test_features, test_labels = extract_features(feature_extractor, test_loader, device)
    
    # Train logistic regression
    print("\nTraining Logistic Regression classifier...")
    print("(This may take a few minutes...)")
    
    clf = LogisticRegression(
        max_iter=args.max_iter,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(train_features, train_labels)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_pred = clf.predict(val_features)
    val_proba = clf.predict_proba(val_features)
    
    val_top1 = accuracy_score(val_labels, val_pred) * 100
    val_top5 = top_k_accuracy_score(val_labels, val_proba, k=5) * 100
    
    print(f"Val Top-1 Accuracy: {val_top1:.2f}%")
    print(f"Val Top-5 Accuracy: {val_top5:.2f}%")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_pred = clf.predict(test_features)
    test_proba = clf.predict_proba(test_features)
    
    test_top1 = accuracy_score(test_labels, test_pred) * 100
    test_top5 = top_k_accuracy_score(test_labels, test_proba, k=5) * 100
    
    print(f"\n{'='*50}")
    print("BASELINE RESULTS (Logistic Regression on Frozen ResNet50)")
    print(f"{'='*50}")
    print(f"Test Top-1 Accuracy: {test_top1:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5:.2f}%")
    print(f"{'='*50}")
    
    # Also compute random baseline
    random_top1 = 100 / 101  # 1/num_classes
    random_top5 = 500 / 101  # 5/num_classes (capped at 100)
    random_top5 = min(random_top5, 100)
    
    print(f"\nRandom Guessing Baseline:")
    print(f"  Top-1: {random_top1:.2f}%")
    print(f"  Top-5: {random_top5:.2f}%")
    
    # Save results
    results = {
        "baseline_type": "logistic_regression_frozen_resnet50",
        "val_top1": val_top1,
        "val_top5": val_top5,
        "test_top1": test_top1,
        "test_top5": test_top5,
        "random_baseline_top1": random_top1,
        "random_baseline_top5": random_top5,
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }
    
    results_path = save_dir / "baseline_classifier_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
   


if __name__ == "__main__":
    main()
