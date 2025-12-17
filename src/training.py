"""
Training utilities for wind power forecasting models.
Includes training loop, data preparation, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from tqdm import tqdm


class WindDataset(Dataset):
    """PyTorch Dataset for wind power forecasting."""
    
    def __init__(self,
                 timeseries_df: pd.DataFrame,
                 node_cols: List[str],
                 input_window: int,
                 forecast_horizon: int,
                 feature_cols: Optional[List[str]] = None):
        """
        Initialize wind dataset.
        
        Args:
            timeseries_df: Time series dataframe
            node_cols: List of columns for each node (power columns)
            input_window: Length of input sequence
            forecast_horizon: Length of forecast
            feature_cols: Additional feature columns per node
        """
        self.df = timeseries_df
        self.node_cols = node_cols
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.feature_cols = feature_cols or []
        
        # Create valid sample indices
        self.valid_indices = list(range(
            input_window,
            len(self.df) - forecast_horizon
        ))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            x: Input [num_nodes, input_window, features]
            y: Target [num_nodes, forecast_horizon]
        """
        sample_idx = self.valid_indices[idx]
        
        # Input window
        start_idx = sample_idx - self.input_window
        end_idx = sample_idx
        
        # Target window
        target_start = sample_idx
        target_end = sample_idx + self.forecast_horizon
        
        # Extract node power values
        x_power = self.df[self.node_cols].iloc[start_idx:end_idx].values
        y_power = self.df[self.node_cols].iloc[target_start:target_end].values
        
        # Transpose to [num_nodes, time]
        x_power = x_power.T  # [num_nodes, input_window]
        y_power = y_power.T  # [num_nodes, forecast_horizon]
        
        # Add feature dimension: [num_nodes, input_window, 1]
        x = np.expand_dims(x_power, axis=-1)
        
        # Add more features if specified
        if self.feature_cols:
            additional_features = []
            for feat_col in self.feature_cols:
                feat_vals = self.df[feat_col].iloc[start_idx:end_idx].values
                # Repeat for each node
                feat_vals = np.tile(feat_vals, (len(self.node_cols), 1))
                additional_features.append(np.expand_dims(feat_vals, axis=-1))
            
            # Concatenate: [num_nodes, input_window, num_features]
            x = np.concatenate([x] + additional_features, axis=-1)
        
        return torch.FloatTensor(x), torch.FloatTensor(y_power)


def create_dataloaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       node_cols: List[str],
                       input_window: int,
                       forecast_horizon: int,
                       batch_size: int = 32,
                       feature_cols: Optional[List[str]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders.
    
    Args:
        train_df, val_df, test_df: Split dataframes
        node_cols: Node column names
        input_window: Input sequence length
        forecast_horizon: Forecast length
        batch_size: Batch size
        feature_cols: Additional feature columns
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = WindDataset(train_df, node_cols, input_window, forecast_horizon, feature_cols)
    val_dataset = WindDataset(val_df, node_cols, input_window, forecast_horizon, feature_cols)
    test_dataset = WindDataset(test_df, node_cols, input_window, forecast_horizon, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def train_epoch(model: nn.Module,
               dataloader: DataLoader,
               optimizer: optim.Optimizer,
               criterion: nn.Module,
               device: torch.device,
               edge_indices: Optional[Dict] = None,
               edge_weights: Optional[Dict] = None,
               use_graph: bool = True) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        edge_indices: Edge indices for graph models
        edge_weights: Edge weights for graph models
        use_graph: Whether model uses graph structure
        
    Returns:
        Average epoch loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_graph and edge_indices is not None:
            # Move edges to device once
            if num_batches == 0:
                edge_indices = {k: v.to(device) for k, v in edge_indices.items()}
                if edge_weights:
                    edge_weights = {k: v.to(device) for k, v in edge_weights.items()}
            
            pred = model(x, edge_indices, edge_weights)
        else:
            pred = model(x)
        
        # Compute loss
        loss = criterion(pred, y)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            edge_indices: Optional[Dict] = None,
            edge_weights: Optional[Dict] = None,
            use_graph: bool = True) -> float:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        edge_indices: Edge indices
        edge_weights: Edge weights
        use_graph: Whether model uses graph
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Move edges to device
    if use_graph and edge_indices is not None:
        edge_indices = {k: v.to(device) for k, v in edge_indices.items()}
        if edge_weights:
            edge_weights = {k: v.to(device) for k, v in edge_weights.items()}
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            
            if use_graph and edge_indices is not None:
                pred = model(x, edge_indices, edge_weights)
            else:
                pred = model(x)
            
            loss = criterion(pred, y)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               config: Dict,
               edge_indices: Optional[Dict] = None,
               edge_weights: Optional[Dict] = None,
               checkpoint_dir: Path = Path("checkpoints")) -> Dict:
    """
    Complete training loop with early stopping and checkpointing.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        edge_indices: Edge indices for graph models
        edge_weights: Edge weights
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Training history dictionary
    """
    # Setup
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.get('scheduler_patience', 5),
        verbose=True
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 15)
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if model uses graph
    use_graph = hasattr(model, 'graph_layers') or 'graph' in model.__class__.__name__.lower()
    
    # Training loop
    num_epochs = config.get('num_epochs', 100)
    
    print(f"Training on device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using graph structure: {use_graph}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            edge_indices, edge_weights, use_graph
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion, device,
            edge_indices, edge_weights, use_graph
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / f"{model.__class__.__name__}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"{model.__class__.__name__}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Save final history
    history_path = checkpoint_dir / f"{model.__class__.__name__}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    return history


def load_checkpoint(model: nn.Module,
                   checkpoint_path: Path,
                   device: torch.device) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


if __name__ == "__main__":
    print("Training utilities module")
    print("Import this module to use training functions")
