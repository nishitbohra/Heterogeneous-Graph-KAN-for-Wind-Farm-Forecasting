"""
Kolmogorov-Arnold Network (KAN) layer implementations.
Uses learnable B-spline basis functions for adaptive transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class BSplineBasis(nn.Module):
    """B-spline basis functions for KAN layer."""
    
    def __init__(self, in_features: int, num_basis: int = 5, degree: int = 3):
        """
        Initialize B-spline basis.
        
        Args:
            in_features: Input feature dimension
            num_basis: Number of basis functions per input feature
            degree: Degree of B-spline (typically 3 for cubic)
        """
        super().__init__()
        self.in_features = in_features
        self.num_basis = num_basis
        self.degree = degree
        
        # Learnable knot positions (initialized uniformly)
        self.register_buffer('knots', torch.linspace(-1, 1, num_basis + degree + 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis function activations.
        
        Args:
            x: Input tensor of shape [batch, in_features]
            
        Returns:
            Basis activations of shape [batch, in_features, num_basis]
        """
        # Normalize input to [-1, 1] range
        x_norm = torch.tanh(x)  # shape: [batch, in_features]
        
        # Compute B-spline basis for each input feature
        batch_size = x_norm.shape[0]
        basis_values = torch.zeros(batch_size, self.in_features, self.num_basis, 
                                   device=x.device, dtype=x.dtype)
        
        for i in range(self.num_basis):
            # Simple B-spline approximation using RBF-like functions
            center = self.knots[i + self.degree // 2]
            width = (self.knots[i + 1] - self.knots[i]) + 1e-6
            
            # Gaussian-like basis function
            diff = (x_norm.unsqueeze(-1) - center) / width
            basis_values[:, :, i] = torch.exp(-0.5 * diff.pow(2)).squeeze(-1)
        
        return basis_values


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer.
    
    Implements: y = Σ_{i,j} c_{ij} * φ_j(x_i)
    where φ_j are learnable basis functions and c_{ij} are coefficients.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 num_basis: int = 5,
                 use_bias: bool = True):
        """
        Initialize KAN layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_basis: Number of basis functions per input
            use_bias: Whether to use bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        
        # B-spline basis functions
        self.basis = BSplineBasis(in_features, num_basis)
        
        # Learnable coefficients: [in_features, num_basis, out_features]
        self.coefficients = nn.Parameter(
            torch.randn(in_features, num_basis, out_features) * 0.1
        )
        
        # Optional bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, in_features]
            
        Returns:
            Output tensor of shape [batch, out_features]
        """
        # Compute basis activations: [batch, in_features, num_basis]
        basis_values = self.basis(x)
        
        # Compute output: einsum for efficient matrix multiplication
        # basis_values: [batch, in_features, num_basis]
        # coefficients: [in_features, num_basis, out_features]
        # output: [batch, out_features]
        output = torch.einsum('bik,iko->bo', basis_values, self.coefficients)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class KANLinear(nn.Module):
    """
    KAN layer with residual connection and normalization.
    More stable for deep networks.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_basis: int = 5,
                 dropout: float = 0.1,
                 use_residual: bool = True):
        """
        Initialize KAN linear layer with normalization.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_basis: Number of basis functions
            dropout: Dropout probability
            use_residual: Whether to use residual connection
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual and (in_features == out_features)
        
        # KAN layer
        self.kan = KANLayer(in_features, out_features, num_basis)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_features)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if needed
        if self.use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual and normalization.
        
        Args:
            x: Input tensor [batch, in_features]
            
        Returns:
            Output tensor [batch, out_features]
        """
        # KAN transformation
        out = self.kan(x)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                out = out + self.residual_proj(x)
            else:
                out = out + x
        
        # Normalize and dropout
        out = self.norm(out)
        out = self.dropout(out)
        
        return out


class MultiLayerKAN(nn.Module):
    """
    Multi-layer KAN network (stacked KAN layers).
    """
    
    def __init__(self,
                 in_features: int,
                 hidden_features: List[int],
                 out_features: int,
                 num_basis: int = 5,
                 dropout: float = 0.1):
        """
        Initialize multi-layer KAN.
        
        Args:
            in_features: Input dimension
            hidden_features: List of hidden layer dimensions
            out_features: Output dimension
            num_basis: Number of basis functions per layer
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(KANLinear(in_features, hidden_features[0], num_basis, dropout))
        
        # Hidden layers
        for i in range(len(hidden_features) - 1):
            layers.append(KANLinear(hidden_features[i], hidden_features[i+1], 
                                   num_basis, dropout))
        
        # Output layer (no dropout, no residual)
        layers.append(KANLayer(hidden_features[-1], out_features, num_basis))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor [batch, in_features]
            
        Returns:
            Output tensor [batch, out_features]
        """
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Test KAN layers
    print("Testing KAN layers...")
    
    batch_size = 32
    in_dim = 10
    out_dim = 5
    
    # Test basic KAN layer
    kan = KANLayer(in_dim, out_dim, num_basis=5)
    x = torch.randn(batch_size, in_dim)
    y = kan(x)
    print(f"KANLayer: input {x.shape} -> output {y.shape}")
    
    # Test KAN linear with normalization
    kan_linear = KANLinear(in_dim, out_dim, num_basis=5)
    y = kan_linear(x)
    print(f"KANLinear: input {x.shape} -> output {y.shape}")
    
    # Test multi-layer KAN
    mlkan = MultiLayerKAN(in_dim, [32, 64, 32], out_dim, num_basis=5)
    y = mlkan(x)
    print(f"MultiLayerKAN: input {x.shape} -> output {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in kan.parameters())
    print(f"\nTotal parameters in basic KAN: {total_params:,}")
    
    print("\nKAN layers test passed!")
