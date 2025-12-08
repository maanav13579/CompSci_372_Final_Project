"""
Training script for calorie predictor. Regressor uses frozen ResNet50 with a custom MLP head

"""
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data import get_calorie_dataloaders, get_train_transforms, get_val_transforms
from models import create_regressor, HuberLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Calorie Regressor")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="nutrition5k", 
                        choices=["nutrition5k", "food101_lookup"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--calorie_scale", type=float, default=1000.0)
    
    # Model
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Training
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=10)
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./artifacts/reg_logs")
    parser.add_argument("--save_dir", type=str, default="./artifacts/models")
    parser.add_argument("--exp_name", type=str, default=None)
    
    return parser.parse_args()


class RegressionMetrics:
    """Compute regression metrics."""
    
    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray, scale: float = 1.0) -> Dict[str, float]:
        """
        Compute MAE, RMSE, R², MAPE.
        
        Args:
            pred: Predicted values (normalized)
            target: Target values (normalized)
            scale: Scale factor to convert back to original units
        """
        # Convert to original scale
        pred_orig = pred * scale
        target_orig = target * scale
        
        # MAE
        mae = np.mean(np.abs(pred_orig - target_orig))
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_orig - target_orig) ** 2))
        
        # R²
        ss_res = np.sum((target_orig - pred_orig) ** 2)
        ss_tot = np.sum((target_orig - np.mean(target_orig)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((target_orig - pred_orig) / (target_orig + 1e-8))) * 100
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
        }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    #training loop for single epoch
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        #implemented mixed precision training
        with autocast(enabled=use_amp):
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        #loss calculations
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        
        pbar.set_postfix({"loss": loss.item()})
    
    return running_loss / total


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    calorie_scale: float,
) -> Tuple[float, Dict[str, float]]:
    """Validate model and compute metrics."""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    #validation loop
    for images, targets in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
        #predict and calculate loss
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    val_loss = running_loss / len(all_preds)
    
    # Compute metrics
    metrics = RegressionMetrics.compute(
        np.array(all_preds),
        np.array(all_targets),
        scale=calorie_scale
    )
    
    return val_loss, metrics


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experment for logs
    if args.exp_name is None:
        args.exp_name = f"regressor_{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Directories
    log_dir = Path(args.log_dir) / args.exp_name
    save_dir = Path(args.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_calorie_dataloaders(
        dataset_type=args.dataset_type,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model
    model = create_regressor(
        backbone=args.backbone,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss
    if args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = HuberLoss(delta=args.huber_delta)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler: Reduce learning rate durng plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    best_mae = float("inf")
    patience_counter = 0
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae': [],
        'rmse': [],
        'r2': [],
        'mape': [],
        'lr': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.use_amp
        )
        
        # Validate
        val_loss, metrics = validate(
            model, val_loader, criterion, device, args.calorie_scale
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {metrics['mae']:.1f} kcal, RMSE: {metrics['rmse']:.1f} kcal")
        print(f"Val R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.1f}%")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(metrics['mae'])
        history['rmse'].append(metrics['rmse'])
        history['r2'].append(metrics['r2'])
        history['mape'].append(metrics['mape'])
        history['lr'].append(current_lr)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/MAE", metrics["mae"], epoch)
        writer.add_scalar("Metrics/RMSE", metrics["rmse"], epoch)
        writer.add_scalar("Metrics/R2", metrics["r2"], epoch)
        writer.add_scalar("Metrics/MAPE", metrics["mape"], epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": vars(args),
            }
            torch.save(checkpoint, save_dir / "regressor_best.pth")
            print(f"Saved best model with MAE: {metrics['mae']:.1f} kcal")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(save_dir / "regressor_best.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_metrics = validate(
        model, test_loader, criterion, device, args.calorie_scale
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAE: {test_metrics['mae']:.1f} kcal")
    print(f"  RMSE: {test_metrics['rmse']:.1f} kcal")
    print(f"  R²: {test_metrics['r2']:.3f}")
    print(f"  MAPE: {test_metrics['mape']:.1f}%")
    
    # Save results
    results = {
        "best_val_mae": float(best_mae),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_r2": float(test_metrics["r2"]),
        "test_mape": float(test_metrics["mape"]),
    }
    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves. Below code is generated using AI-Assistant
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MAE over epochs
    axes[0, 1].plot(epochs_range, history['mae'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (kcal)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: RMSE over epochs
    axes[0, 2].plot(epochs_range, history['rmse'], 'm-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('RMSE (kcal)')
    axes[0, 2].set_title('Root Mean Square Error')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: R² over epochs
    axes[1, 0].plot(epochs_range, history['r2'], 'c-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('R² Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: MAPE over epochs
    axes[1, 1].plot(epochs_range, history['mape'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].set_title('Mean Absolute Percentage Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Learning rate
    axes[1, 2].plot(epochs_range, history['lr'], 'k-', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = Path(args.save_dir).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"regressor_training_curves_{args.exp_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {fig_path}")
    
    writer.close()
    print(f"\nTraining complete! Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
