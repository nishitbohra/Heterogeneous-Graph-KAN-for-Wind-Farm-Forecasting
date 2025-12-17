"""
Heterogeneous Graph-KAN (HG-KAN) model for wind power forecasting.
Combines heterogeneous graph convolution with KAN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .kan_layers import KANLayer, KANLinear, MultiLayerKAN


class HeterogeneousGraphConv(nn.Module):
    """
    Heterogeneous graph convolution layer.
    Processes multiple edge types separately then aggregates.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 edge_types: List[str],
                 aggr: str = 'mean'):
        """
        Initialize heterogeneous graph convolution.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            edge_types: List of edge type names
            aggr: Aggregation method ('mean', 'sum', 'max')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_types = edge_types
        self.aggr = aggr
        
        # Separate transformation for each edge type
        self.edge_transforms = nn.ModuleDict({
            edge_type: nn.Linear(in_features, out_features)
            for edge_type in edge_types
        })
        
        # Self-loop transformation
        self.self_transform = nn.Linear(in_features, out_features)
        
        # Edge type attention (learnable weights for combining edge types)
        self.edge_attention = nn.Parameter(torch.ones(len(edge_types)))
    
    def forward(self, 
                x: torch.Tensor,
                edge_indices: Dict[str, torch.Tensor],
                edge_weights: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_indices: Dict of edge_index tensors [2, num_edges] for each type
            edge_weights: Optional dict of edge weights [num_edges] for each type
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.shape[0]
        
        # Self-loop contribution
        out = self.self_transform(x)
        
        # Process each edge type
        edge_contributions = []
        
        for idx, edge_type in enumerate(self.edge_types):
            if edge_type not in edge_indices:
                continue
                
            edge_index = edge_indices[edge_type]
            if edge_index.shape[1] == 0:  # No edges of this type
                continue
            
            # Transform node features
            x_transformed = self.edge_transforms[edge_type](x)
            
            # Message passing
            src, dst = edge_index[0], edge_index[1]
            messages = x_transformed[src]  # [num_edges, out_features]
            
            # Apply edge weights if provided
            if edge_weights is not None and edge_type in edge_weights:
                weights = edge_weights[edge_type].unsqueeze(-1)  # [num_edges, 1]
                messages = messages * weights
            
            # Aggregate messages for each node
            aggregated = torch.zeros(num_nodes, self.out_features, 
                                    device=x.device, dtype=x.dtype)
            
            if self.aggr == 'mean':
                # Count degree for each node
                degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
                degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
                degree = degree.clamp(min=1).unsqueeze(-1)
                
                aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
                aggregated = aggregated / degree
                
            elif self.aggr == 'sum':
                aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
                
            elif self.aggr == 'max':
                aggregated.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(messages), 
                                          messages, reduce='amax', include_self=False)
            
            # Weight by edge type attention
            attention_weight = torch.softmax(self.edge_attention, dim=0)[idx]
            edge_contributions.append(aggregated * attention_weight)
        
        # Combine all edge type contributions
        if edge_contributions:
            out = out + sum(edge_contributions)
        
        return out


class TemporalConvModule(nn.Module):
    """Temporal convolution module for time series features."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 num_layers: int = 2):
        """
        Initialize temporal convolution.
        
        Args:
            in_channels: Input channels (features)
            out_channels: Output channels
            kernel_size: Convolution kernel size
            num_layers: Number of conv layers
        """
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv1d(in_ch, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, features]
            
        Returns:
            Output [batch, seq_len, out_channels]
        """
        # Conv1d expects [batch, features, seq_len]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class TemporalAttention(nn.Module):
    """Multi-head self-attention for temporal modeling."""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize temporal attention.
        
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, d_model]
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x


class HeterogeneousGraphKAN(nn.Module):
    """
    Complete Heterogeneous Graph-KAN model for wind forecasting.
    
    Architecture:
    1. KAN encoder: Process input features
    2. HG-Conv layers: Heterogeneous graph convolution with spatial message passing
    3. Temporal module: Conv1D or attention for temporal patterns
    4. KAN decoder: Generate multi-step forecasts
    """
    
    def __init__(self,
                 num_nodes: int,
                 input_window: int,
                 forecast_horizon: int,
                 node_features: int,
                 edge_types: List[str],
                 hidden_dim: int = 64,
                 kan_basis: int = 5,
                 num_graph_layers: int = 2,
                 temporal_type: str = 'conv',
                 dropout: float = 0.1):
        """
        Initialize HG-KAN model.
        
        Args:
            num_nodes: Number of nodes in graph
            input_window: Length of input time window
            forecast_horizon: Number of steps to forecast
            node_features: Number of features per node per timestep
            edge_types: List of edge type names
            hidden_dim: Hidden dimension size
            kan_basis: Number of basis functions for KAN layers
            num_graph_layers: Number of graph convolution layers
            temporal_type: 'conv' or 'attention'
            dropout: Dropout probability
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # 1. KAN Encoder: [input_window * node_features] -> [hidden_dim]
        self.encoder = MultiLayerKAN(
            in_features=input_window * node_features,
            hidden_features=[hidden_dim * 2, hidden_dim],
            out_features=hidden_dim,
            num_basis=kan_basis,
            dropout=dropout
        )
        
        # 2. Heterogeneous Graph Convolution Layers
        self.graph_layers = nn.ModuleList([
            HeterogeneousGraphConv(
                in_features=hidden_dim,
                out_features=hidden_dim,
                edge_types=edge_types,
                aggr='mean'
            )
            for _ in range(num_graph_layers)
        ])
        
        self.graph_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_graph_layers)
        ])
        
        # 3. Temporal Module
        self.temporal_type = temporal_type
        if temporal_type == 'conv':
            self.temporal = TemporalConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                num_layers=2
            )
        elif temporal_type == 'attention':
            self.temporal = TemporalAttention(
                d_model=hidden_dim,
                num_heads=4,
                dropout=dropout
            )
        
        # 4. KAN Decoder: [hidden_dim] -> [forecast_horizon]
        self.decoder = MultiLayerKAN(
            in_features=hidden_dim,
            hidden_features=[hidden_dim, hidden_dim // 2],
            out_features=forecast_horizon,
            num_basis=kan_basis,
            dropout=dropout
        )
    
    def forward(self,
                x: torch.Tensor,
                edge_indices: Dict[str, torch.Tensor],
                edge_weights: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input node features [batch, num_nodes, input_window, node_features]
            edge_indices: Dict of edge indices for each edge type
            edge_weights: Optional dict of edge weights
            
        Returns:
            Predictions [batch, num_nodes, forecast_horizon]
        """
        batch_size, num_nodes, input_window, node_features = x.shape
        
        # Flatten temporal and feature dimensions for encoding
        # [batch, num_nodes, input_window * node_features]
        x_flat = x.reshape(batch_size, num_nodes, -1)
        
        # 1. KAN Encoding
        # [batch * num_nodes, input_window * node_features]
        x_encoded = x_flat.reshape(-1, input_window * node_features)
        x_encoded = self.encoder(x_encoded)  # [batch * num_nodes, hidden_dim]
        
        # Reshape back to [batch, num_nodes, hidden_dim]
        x_encoded = x_encoded.reshape(batch_size, num_nodes, self.hidden_dim)
        
        # 2. Graph Convolution Layers
        for graph_layer, norm in zip(self.graph_layers, self.graph_norms):
            # Process each batch sample separately
            graph_outputs = []
            for b in range(batch_size):
                h = graph_layer(x_encoded[b], edge_indices, edge_weights)
                graph_outputs.append(h)
            
            x_graph = torch.stack(graph_outputs, dim=0)  # [batch, num_nodes, hidden_dim]
            
            # Residual connection and normalization
            x_encoded = x_encoded + x_graph
            x_encoded = norm(x_encoded)
        
        # 3. Temporal Module
        # Create temporal sequence: [batch * num_nodes, 1, hidden_dim]
        # For simplicity, we expand to small sequence
        x_temporal = x_encoded.reshape(-1, 1, self.hidden_dim)
        x_temporal = x_temporal.repeat(1, 8, 1)  # [batch * num_nodes, 8, hidden_dim]
        
        if self.temporal_type == 'conv':
            x_temporal = self.temporal(x_temporal)
        elif self.temporal_type == 'attention':
            x_temporal = self.temporal(x_temporal)
        
        # Take mean over temporal dimension
        x_temporal = x_temporal.mean(dim=1)  # [batch * num_nodes, hidden_dim]
        
        # 4. KAN Decoding
        predictions = self.decoder(x_temporal)  # [batch * num_nodes, forecast_horizon]
        
        # Reshape to [batch, num_nodes, forecast_horizon]
        predictions = predictions.reshape(batch_size, num_nodes, self.forecast_horizon)
        
        return predictions


if __name__ == "__main__":
    # Test HG-KAN model
    print("Testing Heterogeneous Graph-KAN model...")
    
    # Model parameters
    num_nodes = 203
    input_window = 24
    forecast_horizon = 6
    node_features = 3  # e.g., speed, power, direction
    edge_types = ['spatial', 'wake', 'correlation']
    batch_size = 8
    
    # Create model
    model = HeterogeneousGraphKAN(
        num_nodes=num_nodes,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        node_features=node_features,
        edge_types=edge_types,
        hidden_dim=64,
        kan_basis=5,
        num_graph_layers=2,
        temporal_type='conv',
        dropout=0.1
    )
    
    # Create dummy data
    x = torch.randn(batch_size, num_nodes, input_window, node_features)
    
    # Create dummy edges
    edge_indices = {
        'spatial': torch.randint(0, num_nodes, (2, 500)),
        'wake': torch.randint(0, num_nodes, (2, 300)),
        'correlation': torch.randint(0, num_nodes, (2, 400))
    }
    
    # Forward pass
    predictions = model(x, edge_indices)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Expected output shape: [batch={batch_size}, nodes={num_nodes}, horizon={forecast_horizon}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nHG-KAN model test passed!")
