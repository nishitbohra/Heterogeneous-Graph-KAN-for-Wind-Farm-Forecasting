"""
Evaluation utilities for wind power forecasting models.
Computes metrics and generates visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute forecasting metrics.
    
    Args:
        y_true: Ground truth [num_samples, num_nodes, horizon]
        y_pred: Predictions [num_samples, num_nodes, horizon]
        
    Returns:
        Dictionary of metrics
    """
    # Flatten to [num_samples * num_nodes * horizon]
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    # Compute metrics
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    
    # MAPE (avoid division by zero)
    mask_nonzero = y_true_flat != 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true_flat[mask_nonzero] - y_pred_flat[mask_nonzero]) / 
                              y_true_flat[mask_nonzero])) * 100
    else:
        mape = np.nan
    
    # R-squared
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Normalized RMSE (by mean of true values)
    nrmse = rmse / (y_true_flat.mean() + 1e-8)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'NRMSE': nrmse
    }


def compute_metrics_per_horizon(y_true: np.ndarray, 
                                y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute metrics for each forecast horizon step.
    
    Args:
        y_true: Ground truth [num_samples, num_nodes, horizon]
        y_pred: Predictions [num_samples, num_nodes, horizon]
        
    Returns:
        DataFrame with metrics per horizon
    """
    horizon = y_true.shape[2]
    
    metrics_list = []
    for h in range(horizon):
        y_true_h = y_true[:, :, h]
        y_pred_h = y_pred[:, :, h]
        
        metrics = compute_metrics(
            y_true_h.reshape(-1, 1, 1),
            y_pred_h.reshape(-1, 1, 1)
        )
        metrics['horizon'] = h + 1
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def evaluate_model(model: nn.Module,
                  dataloader,
                  device: torch.device,
                  edge_indices: Optional[Dict] = None,
                  edge_weights: Optional[Dict] = None,
                  use_graph: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model and return predictions.
    
    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device
        edge_indices: Edge indices
        edge_weights: Edge weights
        use_graph: Whether model uses graph
        
    Returns:
        y_true, y_pred arrays
    """
    model.eval()
    
    # Move edges to device
    if use_graph and edge_indices is not None:
        edge_indices = {k: v.to(device) for k, v in edge_indices.items()}
        if edge_weights:
            edge_weights = {k: v.to(device) for k, v in edge_weights.items()}
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            
            if use_graph and edge_indices is not None:
                pred = model(x, edge_indices, edge_weights)
            else:
                pred = model(x)
            
            predictions.append(pred.cpu().numpy())
            targets.append(y.numpy())
    
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    
    return y_true, y_pred


def plot_training_history(history: Dict, save_path: Optional[Path] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1].plot(history['lr'], linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    node_idx: int = 0,
                    num_samples: int = 100,
                    save_path: Optional[Path] = None):
    """
    Plot predictions vs ground truth for a specific node.
    
    Args:
        y_true: Ground truth [num_samples, num_nodes, horizon]
        y_pred: Predictions [num_samples, num_nodes, horizon]
        node_idx: Node index to plot
        num_samples: Number of samples to plot
        save_path: Path to save figure
    """
    horizon = y_true.shape[2]
    
    fig, axes = plt.subplots(1, horizon, figsize=(4*horizon, 4))
    if horizon == 1:
        axes = [axes]
    
    for h in range(horizon):
        ax = axes[h]
        
        y_t = y_true[:num_samples, node_idx, h]
        y_p = y_pred[:num_samples, node_idx, h]
        
        ax.plot(y_t, label='Ground Truth', alpha=0.7, linewidth=2)
        ax.plot(y_p, label='Prediction', alpha=0.7, linewidth=2)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Power Output')
        ax.set_title(f'Horizon +{h+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Predictions for Node {node_idx}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")
    
    plt.show()


def plot_scatter(y_true: np.ndarray,
                y_pred: np.ndarray,
                save_path: Optional[Path] = None):
    """
    Plot scatter plot of predictions vs ground truth.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        save_path: Path to save figure
    """
    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaNs
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    # Subsample for plotting
    if len(y_true_flat) > 10000:
        indices = np.random.choice(len(y_true_flat), 10000, replace=False)
        y_true_flat = y_true_flat[indices]
        y_pred_flat = y_pred_flat[indices]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Metrics
    metrics = compute_metrics(
        y_true.reshape(-1, 1, 1),
        y_pred.reshape(-1, 1, 1)
    )
    
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title('Prediction vs Ground Truth', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics_text = f"MAE: {metrics['MAE']:.3f}\nRMSE: {metrics['RMSE']:.3f}\nR²: {metrics['R2']:.3f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to {save_path}")
    
    plt.show()


def plot_metrics_per_horizon(metrics_df: pd.DataFrame,
                            save_path: Optional[Path] = None):
    """
    Plot metrics for each forecast horizon.
    
    Args:
        metrics_df: DataFrame with metrics per horizon
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'R2']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        if metric in metrics_df.columns:
            ax.plot(metrics_df['horizon'], metrics_df[metric], 
                   marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Forecast Horizon')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs Forecast Horizon')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(metrics_df['horizon'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")
    
    plt.show()


def compare_models(results: Dict[str, Dict],
                  save_path: Optional[Path] = None):
    """
    Compare multiple models side by side.
    
    Args:
        results: Dict of {model_name: metrics_dict}
        save_path: Path to save figure
    """
    # Prepare data
    models = list(results.keys())
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = [results[model].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Rotate x labels if too many models
        if len(models) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def save_results(model_name: str,
                metrics: Dict,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                save_dir: Path):
    """
    Save evaluation results.
    
    Args:
        model_name: Name of the model
        metrics: Metrics dictionary
        y_true: Ground truth
        y_pred: Predictions
        save_dir: Directory to save results
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    import json
    
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in metrics.items()}
    
    metrics_path = save_dir / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save predictions
    np.savez_compressed(
        save_dir / f"{model_name}_predictions.npz",
        y_true=y_true,
        y_pred=y_pred
    )
    
    print(f"Results saved to {save_dir}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    print("Evaluation utilities module")
    print("Import this module to use evaluation functions")
