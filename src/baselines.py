"""
Baseline models for wind power forecasting comparison.
Includes: Persistence, LSTM, GCN-GRU, and ST-GAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PersistenceModel(nn.Module):
    """
    Persistence (naive) baseline.
    Predicts that future values will equal the last observed value.
    """
    
    def __init__(self, forecast_horizon: int = 1):
        """
        Initialize persistence model.
        
        Args:
            forecast_horizon: Number of steps to forecast
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - repeat last timestep.
        
        Args:
            x: Input [batch, num_nodes, input_window, features]
            
        Returns:
            Predictions [batch, num_nodes, forecast_horizon]
        """
        # Take last timestep of first feature (power)
        last_value = x[:, :, -1, 0]  # [batch, num_nodes]
        
        # Repeat for forecast horizon
        predictions = last_value.unsqueeze(-1).repeat(1, 1, self.forecast_horizon)
        
        return predictions


class LSTMBaseline(nn.Module):
    """
    LSTM baseline without graph structure.
    Processes each node independently.
    """
    
    def __init__(self,
                 input_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 forecast_horizon: int = 1,
                 dropout: float = 0.1):
        """
        Initialize LSTM baseline.
        
        Args:
            input_features: Number of input features per timestep
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            forecast_horizon: Number of steps to forecast
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, forecast_horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, num_nodes, input_window, features]
            
        Returns:
            Predictions [batch, num_nodes, forecast_horizon]
        """
        batch_size, num_nodes, seq_len, features = x.shape
        
        # Reshape: [batch * num_nodes, seq_len, features]
        x = x.reshape(-1, seq_len, features)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch * num_nodes, hidden_dim]
        
        # Project to forecast horizon
        predictions = self.fc(last_hidden)  # [batch * num_nodes, forecast_horizon]
        
        # Reshape back: [batch, num_nodes, forecast_horizon]
        predictions = predictions.reshape(batch_size, num_nodes, self.forecast_horizon)
        
        return predictions


class GraphConvLSTM(nn.Module):
    """
    GCN-GRU/LSTM baseline.
    Combines graph convolution with LSTM for spatio-temporal modeling.
    """
    
    def __init__(self,
                 input_features: int,
                 hidden_dim: int = 64,
                 num_graph_layers: int = 2,
                 forecast_horizon: int = 1,
                 dropout: float = 0.1):
        """
        Initialize GCN-LSTM model.
        
        Args:
            input_features: Number of input features
            hidden_dim: Hidden dimension
            num_graph_layers: Number of graph conv layers
            forecast_horizon: Forecast steps
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_features, hidden_dim)
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_graph_layers)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
    
    def graph_conv(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Simple graph convolution.
        
        Args:
            x: Node features [num_nodes, features]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated features [num_nodes, features]
        """
        num_nodes = x.shape[0]
        
        # Aggregate neighbor features
        src, dst = edge_index[0], edge_index[1]
        messages = x[src]
        
        # Sum aggregation
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        
        # Normalize by degree
        degree = torch.zeros(num_nodes, device=x.device)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        degree = degree.clamp(min=1).unsqueeze(-1)
        aggregated = aggregated / degree
        
        return aggregated
    
    def forward(self,
                x: torch.Tensor,
                edge_indices: dict = None,
                edge_weights: dict = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, num_nodes, input_window, features]
            edge_indices: Dict of edge indices (uses 'spatial' key)
            edge_weights: Dict of edge weights (unused)
            
        Returns:
            Predictions [batch, num_nodes, forecast_horizon]
        """
        batch_size, num_nodes, seq_len, features = x.shape
        
        # Extract spatial edge index
        if edge_indices is not None and 'spatial' in edge_indices:
            edge_index = edge_indices['spatial']
        else:
            # If no edges provided, create empty tensor
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        
        # Process each timestep with graph conv
        temporal_features = []
        
        for t in range(seq_len):
            x_t = x[:, :, t, :]  # [batch, num_nodes, features]
            
            # Project features
            h_t = self.input_proj(x_t)  # [batch, num_nodes, hidden_dim]
            
            # Apply graph convolutions (per batch)
            for graph_conv in self.graph_convs:
                h_list = []
                for b in range(batch_size):
                    h_b = self.graph_conv(h_t[b], edge_index)
                    h_b = graph_conv(h_b)
                    h_b = F.relu(h_b)
                    h_list.append(h_b)
                h_t = torch.stack(h_list, dim=0)
            
            temporal_features.append(h_t)
        
        # Stack temporal features: [batch, num_nodes, seq_len, hidden_dim]
        temporal_features = torch.stack(temporal_features, dim=2)
        
        # Reshape for LSTM: [batch * num_nodes, seq_len, hidden_dim]
        temporal_features = temporal_features.reshape(-1, seq_len, self.hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(temporal_features)
        last_hidden = lstm_out[:, -1, :]  # [batch * num_nodes, hidden_dim]
        
        # Decode
        predictions = self.decoder(last_hidden)  # [batch * num_nodes, forecast_horizon]
        
        # Reshape: [batch, num_nodes, forecast_horizon]
        predictions = predictions.reshape(batch_size, num_nodes, self.forecast_horizon)
        
        return predictions


class STGATLayer(nn.Module):
    """Spatio-Temporal Graph Attention Layer."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        """
        Initialize ST-GAT layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        
        # Output projection
        self.out_proj = nn.Linear(out_features, out_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with graph attention.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated features [num_nodes, out_features]
        """
        num_nodes = x.shape[0]
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores for edges
        src, dst = edge_index[0], edge_index[1]
        
        # Attention: Q[dst] * K[src]
        attn_scores = (Q[dst] * K[src]).sum(dim=-1) / (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=0)  # [num_edges, num_heads]
        
        # Apply attention to values
        messages = V[src] * attn_scores.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
        
        # Aggregate
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, 
                         device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(messages), messages)
        
        # Reshape and project
        out = out.reshape(num_nodes, self.out_features)
        out = self.out_proj(out)
        
        return out


class STGATBaseline(nn.Module):
    """Spatio-Temporal Graph Attention Network baseline."""
    
    def __init__(self,
                 input_features: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 forecast_horizon: int = 1,
                 dropout: float = 0.1):
        """
        Initialize ST-GAT model.
        
        Args:
            input_features: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of ST-GAT layers
            forecast_horizon: Forecast horizon
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_features, hidden_dim)
        
        # ST-GAT layers
        self.gat_layers = nn.ModuleList([
            STGATLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor, edge_indices: dict = None, edge_weights: dict = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, num_nodes, input_window, features]
            edge_indices: Dict of edge indices (uses 'spatial' key)
            edge_weights: Dict of edge weights (unused)
            
        Returns:
            Predictions [batch, num_nodes, forecast_horizon]
        """
        batch_size, num_nodes, seq_len, features = x.shape
        
        # Extract spatial edge index
        if edge_indices is not None and 'spatial' in edge_indices:
            edge_index = edge_indices['spatial']
        else:
            # If no edges provided, create empty tensor
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        
        # Input projection
        x = x.reshape(-1, seq_len, features)
        x = self.input_proj(x)  # [batch * num_nodes, seq_len, hidden_dim]
        
        # Temporal attention
        x_temporal, _ = self.temporal_attn(x, x, x)
        x = x + x_temporal  # Residual
        
        # Take last timestep
        x = x[:, -1, :].reshape(batch_size, num_nodes, self.hidden_dim)
        
        # Apply spatial GAT layers
        for gat_layer in self.gat_layers:
            x_list = []
            for b in range(batch_size):
                h = gat_layer(x[b], edge_index)
                x_list.append(h)
            x_spatial = torch.stack(x_list, dim=0)
            x = x + x_spatial  # Residual
        
        # Decode
        x = x.reshape(-1, self.hidden_dim)
        predictions = self.decoder(x)
        predictions = predictions.reshape(batch_size, num_nodes, self.forecast_horizon)
        
        return predictions


if __name__ == "__main__":
    print("Testing baseline models...")
    
    batch_size = 4
    num_nodes = 203
    input_window = 24
    forecast_horizon = 6
    node_features = 3
    
    # Dummy input
    x = torch.randn(batch_size, num_nodes, input_window, node_features)
    edge_index = torch.randint(0, num_nodes, (2, 500))
    
    # Test Persistence
    model = PersistenceModel(forecast_horizon)
    out = model(x)
    print(f"Persistence: {x.shape} -> {out.shape}")
    
    # Test LSTM
    model = LSTMBaseline(node_features, hidden_dim=64, forecast_horizon=forecast_horizon)
    out = model(x)
    print(f"LSTM: {x.shape} -> {out.shape}")
    
    # Test GCN-LSTM
    model = GraphConvLSTM(node_features, hidden_dim=64, forecast_horizon=forecast_horizon)
    out = model(x, edge_index)
    print(f"GCN-LSTM: {x.shape} -> {out.shape}")
    
    # Test ST-GAT
    model = STGATBaseline(node_features, hidden_dim=64, forecast_horizon=forecast_horizon)
    out = model(x, edge_index)
    print(f"ST-GAT: {x.shape} -> {out.shape}")
    
    print("\nAll baseline models tested successfully!")
