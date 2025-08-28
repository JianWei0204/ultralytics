# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Fallback implementation for causal conv1d operation when CUDA extensions are not available.

This module provides pure PyTorch implementations of causal 1D convolution operations
that are typically implemented in CUDA for performance. The fallback ensures functionality
when the optimized CUDA extensions cannot be compiled or are not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Pure PyTorch fallback implementation of causal 1D convolution.
    
    Performs a 1D convolution that only looks at past timesteps (causal).
    
    Args:
        x: Input tensor of shape (batch, channels, seq_len)
        weight: Convolution weight of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        activation: Optional activation function ('silu', 'swish', or None)
        
    Returns:
        Output tensor of shape (batch, out_channels, seq_len)
    """
    batch, in_channels, seq_len = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Causal padding: pad on the left with (kernel_size - 1) zeros
    padding = kernel_size - 1
    x_padded = F.pad(x, (padding, 0))
    
    # Perform 1D convolution
    out = F.conv1d(x_padded, weight, bias)
    
    # Apply activation if specified
    if activation in ['silu', 'swish']:
        out = F.silu(out)
    
    return out


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update function for causal conv1d with state.
    
    Args:
        x: Input tensor of shape (batch, channels)
        conv_state: Convolution state of shape (batch, channels, kernel_size-1)
        weight: Convolution weight of shape (out_channels, in_channels, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        activation: Optional activation function ('silu', 'swish', or None)
        
    Returns:
        Tuple of (output, new_conv_state)
    """
    batch, in_channels = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Update conv_state: shift left and add new input
    if kernel_size > 1:
        new_conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        # Concatenate state and current input for convolution
        conv_input = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)
    else:
        new_conv_state = conv_state
        conv_input = x.unsqueeze(-1)
    
    # Perform convolution
    out = F.conv1d(conv_input, weight, bias)
    out = out.squeeze(-1)  # Remove seq_len dimension
    
    # Apply activation if specified
    if activation in ['silu', 'swish']:
        out = F.silu(out)
    
    return out, new_conv_state


class CausalConv1dFallback(nn.Module):
    """
    Fallback implementation of causal 1D convolution that mimics the CUDA interface.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: Optional[str] = None,
    ):
        """
        Initialize causal 1D convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            kernel_size: Size of convolution kernel
            bias: Whether to use bias
            activation: Optional activation function
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return causal_conv1d_fn(x, self.weight, self.bias, self.activation)
    
    def step(self, x: torch.Tensor, conv_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single step forward with state update."""
        return causal_conv1d_update(x, conv_state, self.weight, self.bias, self.activation)


# Compatibility aliases to match the expected CUDA interface
def causal_conv1d_cuda(*args, **kwargs):
    """Compatibility function that redirects to fallback implementation."""
    warnings.warn(
        "causal_conv1d_cuda not available, using PyTorch fallback. "
        "Performance will be slower than the CUDA implementation.",
        UserWarning
    )
    return causal_conv1d_fn(*args, **kwargs)


def causal_conv1d_update_cuda(*args, **kwargs):
    """Compatibility function for update that redirects to fallback implementation."""
    return causal_conv1d_update(*args, **kwargs)


# Warn user that fallback is being used  
warnings.warn(
    "Using PyTorch fallback for causal_conv1d_cuda. "
    "Performance will be significantly slower than the CUDA implementation. "
    "Consider installing causal-conv1d with CUDA support for better performance.",
    UserWarning
)