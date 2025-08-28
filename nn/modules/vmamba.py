# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Vision Mamba (VMamba) implementation with fallback support for missing CUDA extensions.

This module implements VMamba blocks that can work both with optimized CUDA extensions
and pure PyTorch fallbacks when CUDA extensions are not available.
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA extensions, fall back to PyTorch implementations
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    # Import our fallback implementation
    from .selective_scan_fallback import selective_scan_fn, SelectiveScanFallback
    mamba_inner_fn = None
    warnings.warn(
        "mamba_ssm not available, using PyTorch fallback implementation. "
        "Install mamba_ssm for better performance: pip install mamba-ssm",
        UserWarning
    )

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    CAUSAL_CONV1D_AVAILABLE = False
    # Import our fallback implementation
    from .causal_conv1d_fallback import causal_conv1d_fn, causal_conv1d_update
    warnings.warn(
        "causal_conv1d not available, using PyTorch fallback implementation. "
        "Install causal-conv1d for better performance: pip install causal-conv1d",
        UserWarning
    )


class Mamba(nn.Module):
    """
    Mamba block implementation with fallback support.
    
    This is a simplified version of the Mamba block that works with both
    CUDA extensions and pure PyTorch fallbacks.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
    ):
        """
        Initialize Mamba block.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Local convolution width
            expand: Expansion factor
            dt_rank: Rank of dt projection
            dt_min: Minimum dt value
            dt_max: Maximum dt value
            dt_init: dt initialization method
            dt_scale: dt scaling factor
            dt_init_floor: dt initialization floor
            conv_bias: Whether to use bias in convolution
            bias: Whether to use bias in linear layers
            use_fast_path: Whether to use fast path (when available)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path and MAMBA_AVAILABLE
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A parameter (diagonal state matrix)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = hidden_states.shape
        
        # Input projection
        xz = self.in_proj(hidden_states)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each is (batch, seq_len, d_inner)
        
        # Transpose for convolution: (batch, d_inner, seq_len)
        x = x.transpose(1, 2)
        
        # Causal convolution
        if CAUSAL_CONV1D_AVAILABLE and self.use_fast_path:
            x = causal_conv1d_fn(x, self.conv1d.weight.squeeze(1), self.conv1d.bias, "silu")
        else:
            # Fallback: use standard conv1d with causal padding
            x = F.pad(x, (self.d_conv - 1, 0))
            x = self.conv1d(x)[:, :, :seq_len]
            x = F.silu(x)
        
        # Transpose back: (batch, seq_len, d_inner)
        x = x.transpose(1, 2)
        
        # SSM parameters
        x_dbl = self.x_proj(x)  # (batch, seq_len, dt_rank + 2 * d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # dt projection
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        
        # A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Transpose for selective scan: (batch, d_inner, seq_len)
        x = x.transpose(1, 2)
        dt = dt.transpose(1, 2)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)
        
        # Selective scan
        if MAMBA_AVAILABLE and self.use_fast_path and mamba_inner_fn is not None:
            # Use optimized CUDA implementation if available
            y = mamba_inner_fn(
                x, self.conv1d.weight, self.conv1d.bias, dt, A, B, C, self.D.float(), 
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True
            )
        else:
            # Use fallback implementation
            y = selective_scan_fn(
                u=x,
                delta=dt,
                A=A,
                B=B,
                C=C,
                D=self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        
        # Transpose back: (batch, seq_len, d_inner)
        y = y.transpose(1, 2)
        
        # Apply gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class VMambaBlock(nn.Module):
    """
    Vision Mamba block with residual connection and layer norm.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs
    ):
        """
        Initialize VMamba block.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Local convolution width  
            expand: Expansion factor
            norm_layer: Normalization layer
            **kwargs: Additional arguments for Mamba
        """
        super().__init__()
        self.norm = norm_layer(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        return x + self.mamba(self.norm(x))


class VMamba(nn.Module):
    """
    Vision Mamba model consisting of multiple VMamba blocks.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs
    ):
        """
        Initialize VMamba model.
        
        Args:
            d_model: Model dimension
            n_layers: Number of VMamba blocks
            d_state: State dimension
            d_conv: Local convolution width
            expand: Expansion factor
            norm_layer: Normalization layer
            **kwargs: Additional arguments for VMamba blocks
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Stack of VMamba blocks
        self.layers = nn.ModuleList([
            VMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm_layer=norm_layer,
                **kwargs
            )
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm_f = norm_layer(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all VMamba blocks.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        return self.norm_f(x)


# Make key classes available at module level
__all__ = [
    "Mamba",
    "VMambaBlock", 
    "VMamba",
    "selective_scan_fn",
    "causal_conv1d_fn",
    "MAMBA_AVAILABLE",
    "CAUSAL_CONV1D_AVAILABLE",
]