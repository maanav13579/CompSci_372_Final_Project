"""
Baseline comparison for calorie regression. Compares trained regressor (frozen ResNet50 with trained MLP head)
with just guessing the mean calories of the dataset.

"""
import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data import get_calorie_dataloaders
from models import create_regressor


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Regressor Comparison")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained regressor")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--calorie_scale", type=float, default=1000.0)
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    return parser.parse_args()


def compute_metrics(predictions, targets):
    """Compute regression metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # MAPE
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


@torch.no_grad()
def evaluate_regressor(model, dataloader, device, calorie_scale):
    """Evaluate trained regressor on test set."""
    model.eval()
    predictions = []
    targets = []
    
    for images, labels in tqdm(dataloader, desc="Evaluating regressor"):
        images = images.to(device)
        outputs = model(images).squeeze()
        
        # Convert back to original scale
        preds = outputs.cpu().numpy() * calorie_scale
        tgts = labels.numpy() * calorie_scale
        
        # Handle single item batches
        if preds.ndim == 0:
            predictions.append(float(preds))
        else:
            predictions.extend(preds.tolist())
            
        if tgts.ndim == 0:
            targets.append(float(tgts))
        else:
            targets.extend(tgts.tolist())
    
    return predictions, targets


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    fig_dir = save_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    _, _, test_loader = get_calorie_dataloaders(
        dataset_type="nutrition5k",
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load trained regressor
    print(f"\nLoading trained regressor from {args.model_path}...")
    model = create_regressor(backbone="resnet50", freeze_backbone=True)
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Evaluate trained regressor
    print("\nEvaluating trained regressor...")
    reg_predictions, targets = evaluate_regressor(
        model, test_loader, device, args.calorie_scale
    )
    reg_metrics = compute_metrics(reg_predictions, targets)
    
    # Evaluate mean baseline (predict average calorie for all samples)
    print("Evaluating mean baseline...")
    mean_calorie = np.mean(targets)
    mean_predictions = [mean_calorie] * len(targets)
    mean_metrics = compute_metrics(mean_predictions, targets)
    
    # Print results. All the code below is AI generated. It prints and presents the 
    # model comparison data using bar charts and scatter plots.
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"\nTest set statistics:")
    print(f"  Samples: {len(targets)}")
    print(f"  Mean calories: {np.mean(targets):.1f} kcal")
    print(f"  Std calories: {np.std(targets):.1f} kcal")
    print(f"  Min calories: {np.min(targets):.1f} kcal")
    print(f"  Max calories: {np.max(targets):.1f} kcal")
    
    print(f"\n1. Mean Baseline (always predict {mean_calorie:.1f} kcal):")
    print(f"   MAE:  {mean_metrics['mae']:.1f} kcal")
    print(f"   RMSE: {mean_metrics['rmse']:.1f} kcal")
    print(f"   R²:   {mean_metrics['r2']:.3f}")
    print(f"   MAPE: {mean_metrics['mape']:.1f}%")
    
    print(f"\n2. Trained Regressor:")
    print(f"   MAE:  {reg_metrics['mae']:.1f} kcal")
    print(f"   RMSE: {reg_metrics['rmse']:.1f} kcal")
    print(f"   R²:   {reg_metrics['r2']:.3f}")
    print(f"   MAPE: {reg_metrics['mape']:.1f}%")
    
    print(f"\n{'='*60}")
    print("IMPROVEMENT OVER BASELINE")
    print(f"{'='*60}")
    mae_improvement = mean_metrics['mae'] - reg_metrics['mae']
    mae_improvement_pct = (mae_improvement / mean_metrics['mae']) * 100
    print(f"  MAE reduction: {mae_improvement:.1f} kcal ({mae_improvement_pct:.1f}% improvement)")
    
    rmse_improvement = mean_metrics['rmse'] - reg_metrics['rmse']
    rmse_improvement_pct = (rmse_improvement / mean_metrics['rmse']) * 100
    print(f"  RMSE reduction: {rmse_improvement:.1f} kcal ({rmse_improvement_pct:.1f}% improvement)")
    
    r2_improvement = reg_metrics['r2'] - mean_metrics['r2']
    print(f"  R² improvement: {r2_improvement:.3f}")
    print(f"{'='*60}")

    #AI generated code ends here
    
    # Save results
    results = {
        "test_samples": len(targets),
        "mean_calorie": float(mean_calorie),
        "std_calorie": float(np.std(targets)),
        "mean_baseline": mean_metrics,
        "trained_regressor": reg_metrics,
        "mae_improvement_kcal": float(mae_improvement),
        "mae_improvement_pct": float(mae_improvement_pct),
        "rmse_improvement_kcal": float(rmse_improvement),
        "rmse_improvement_pct": float(rmse_improvement_pct),
    }
    
    results_path = save_dir / "baseline_regressor_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MAE and RMSE comparison
    metrics_names = ['MAE', 'RMSE']
    mean_values = [mean_metrics['mae'], mean_metrics['rmse']]
    reg_values = [reg_metrics['mae'], reg_metrics['rmse']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, mean_values, width, label='Mean Baseline', color='gray')
    bars2 = axes[0].bar(x + width/2, reg_values, width, label='Trained Regressor', color='steelblue')
    
    axes[0].set_ylabel('Error (kcal)', fontsize=12)
    axes[0].set_title('Calorie Prediction Error Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        axes[0].annotate(f'{bar.get_height():.0f}', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        axes[0].annotate(f'{bar.get_height():.0f}', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: R² comparison
    r2_values = [mean_metrics['r2'], reg_metrics['r2']]
    model_names = ['Mean Baseline', 'Trained Regressor']
    colors = ['gray', 'steelblue']
    
    bars = axes[1].bar(model_names, r2_values, color=colors)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('R² Score Comparison (higher is better)', fontsize=14)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        axes[1].annotate(f'{bar.get_height():.3f}', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = fig_dir / "regressor_baseline_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {fig_path}")
    
    # Create scatter plot: Predicted vs Actual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot for mean baseline
    axes[0].scatter(targets, mean_predictions, alpha=0.5, color='gray', s=20)
    axes[0].plot([0, max(targets)], [0, max(targets)], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual Calories (kcal)', fontsize=12)
    axes[0].set_ylabel('Predicted Calories (kcal)', fontsize=12)
    axes[0].set_title(f'Mean Baseline (MAE: {mean_metrics["mae"]:.0f} kcal)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot for regressor
    axes[1].scatter(targets, reg_predictions, alpha=0.5, color='steelblue', s=20)
    axes[1].plot([0, max(targets)], [0, max(targets)], 'r--', label='Perfect prediction')
    axes[1].set_xlabel('Actual Calories (kcal)', fontsize=12)
    axes[1].set_ylabel('Predicted Calories (kcal)', fontsize=12)
    axes[1].set_title(f'Trained Regressor (MAE: {reg_metrics["mae"]:.0f} kcal)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    scatter_path = fig_dir / "regressor_scatter_comparison.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
