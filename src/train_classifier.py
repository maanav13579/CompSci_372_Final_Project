"""
Training script for Food-101 classification.

Usage:
    python train_classifier.py --data_root ./data/food101 --epochs 30
"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data import get_food101_dataloaders, get_train_transforms, get_val_transforms
from models import create_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train Food-101 Classifier")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Path to Food-101 dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Model
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "vit_b_16"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pretrained", action="store_true", default=True)
    
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=0.001)
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./artifacts/class_logs")
    parser.add_argument("--save_dir", type=str, default="./artifacts/models")
    parser.add_argument("--exp_name", type=str, default=None)
    
    return parser.parse_args()


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True  # First epoch, save model
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True  # Improved, save model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Not improved


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": loss.item(), "acc": 100. * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate model."""
    model.eval()
    
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for images, labels in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        
        # Top-1 accuracy
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(labels).sum().item()
        
        # Top-5 accuracy
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += top5_pred.eq(labels.view(-1, 1)).any(dim=1).sum().item()
        
        total += labels.size(0)
    
    val_loss = running_loss / total
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total
    
    return val_loss, top1_acc, top5_acc


def get_warmup_lr(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Linear warmup learning rate."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def main():
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment name
    if args.exp_name is None:
        args.exp_name = f"{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directories
    log_dir = Path(args.log_dir) / args.exp_name
    save_dir = Path(args.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_food101_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model
    model = create_classifier(
        num_classes=101,
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    
    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode="max")
    
    # Training loop
    best_acc = 0.0
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_top1': [],
        'val_top5': [],
        'lr': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, args.warmup_epochs, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.grad_clip, args.use_amp
        )
        
        # Validate
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Top-1: {val_top1:.2f}%, Val Top-5: {val_top5:.2f}%")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_top1'].append(val_top1)
        history['val_top5'].append(val_top5)
        history['lr'].append(current_lr)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val_top1", val_top1, epoch)
        writer.add_scalar("Accuracy/val_top5", val_top5, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        # Scheduler step
        if epoch >= args.warmup_epochs:
            if args.scheduler == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_top1)
        
        # Early stopping check and save best model
        is_best = early_stopping(val_top1)
        
        if is_best:
            best_acc = val_top1
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "config": vars(args),
            }
            torch.save(checkpoint, save_dir / f"classifier_best.pth")
            print(f"Saved best model with Top-1: {val_top1:.2f}%")
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(save_dir / "classifier_best.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_top1, test_top5 = validate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Top-1 Accuracy: {test_top1:.2f}%")
    print(f"  Top-5 Accuracy: {test_top5:.2f}%")
    
    # Save final results
    results = {
        "best_val_top1": float(best_acc),
        "test_top1": float(test_top1),
        "test_top5": float(test_top5),
        "test_loss": float(test_loss),
    }
    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_top1'], 'r-', label='Val Top-1', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_top5'], 'g-', label='Val Top-5', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    axes[1, 0].plot(epochs_range, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Train vs Val accuracy gap (overfitting indicator)
    train_val_gap = [t - v for t, v in zip(history['train_acc'], history['val_top1'])]
    axes[1, 1].plot(epochs_range, train_val_gap, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Acc - Val Acc (%)')
    axes[1, 1].set_title('Overfitting Gap (Train - Val Accuracy)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = Path(args.save_dir).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"classifier_training_curves_{args.exp_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {fig_path}")
    
    writer.close()
    print(f"\nTraining complete! Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
