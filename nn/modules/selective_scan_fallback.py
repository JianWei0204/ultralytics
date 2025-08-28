# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Fallback implementation for selective scan operation when CUDA extensions are not available.

This module provides pure PyTorch implementations of selective scan operations that are
typically implemented in CUDA for performance. The fallback ensures functionality when
the optimized CUDA extensions cannot be compiled or are not available.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor, 
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> torch.Tensor:
    """
    Pure PyTorch fallback implementation of selective scan.
    
    This is a reference implementation that provides the same API as the CUDA version
    but runs on CPU/GPU using standard PyTorch operations.
    
    Args:
        u: Input tensor of shape (batch, dim, seq_len)
        delta: Delta tensor of shape (batch, dim, seq_len)  
        A: A matrix of shape (dim, state_size)
        B: B tensor of shape (batch, state_size, seq_len)
        C: C tensor of shape (batch, state_size, seq_len)
        D: Optional D tensor of shape (dim,)
        z: Optional z tensor of shape (batch, dim, seq_len)
        delta_bias: Optional delta bias of shape (dim,)
        delta_softplus: Whether to apply softplus to delta
        return_last_state: Whether to return the last state
        
    Returns:
        Output tensor of shape (batch, dim, seq_len)
    """
    batch, dim, seq_len = u.shape
    state_size = A.shape[1]
    
    # Apply delta bias if provided
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    
    # Apply softplus to delta if requested
    if delta_softplus:
        delta = F.softplus(delta)
    
    # Initialize state
    device = u.device
    dtype = u.dtype
    x = torch.zeros(batch, dim, state_size, device=device, dtype=dtype)
    
    # Output tensor
    y = torch.zeros_like(u)
    
    # Scan through sequence
    for i in range(seq_len):
        # Get current timestep inputs
        u_i = u[:, :, i]  # (batch, dim)
        delta_i = delta[:, :, i]  # (batch, dim)
        B_i = B[:, :, i]  # (batch, state_size)
        C_i = C[:, :, i]  # (batch, state_size)
        
        # Update state: x = (A * delta).exp() * x + (delta * B * u).unsqueeze(-1)
        # A is (dim, state_size), delta_i is (batch, dim)
        A_delta = A.unsqueeze(0) * delta_i.unsqueeze(-1)  # (batch, dim, state_size)
        x = torch.exp(A_delta) * x + delta_i.unsqueeze(-1) * B_i.unsqueeze(1) * u_i.unsqueeze(-1)
        
        # Output: y = C * x + D * u (if D is provided)
        y_i = torch.sum(C_i.unsqueeze(1) * x, dim=-1)  # (batch, dim)
        if D is not None:
            y_i = y_i + D.unsqueeze(0) * u_i
        
        y[:, :, i] = y_i
    
    # Apply gating if z is provided
    if z is not None:
        y = y * F.silu(z)
    
    if return_last_state:
        return y, x
    return y


class SelectiveScanFallback:
    """
    Fallback class that mimics the interface of the CUDA selective scan implementation.
    """
    
    @staticmethod
    def apply(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor, 
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        delta_bias: Optional[torch.Tensor] = None,
        delta_softplus: bool = False,
        return_last_state: bool = False,
    ) -> torch.Tensor:
        """Apply selective scan with fallback implementation."""
        return selective_scan_fn(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
        )


# Warn user that fallback is being used
warnings.warn(
    "Using PyTorch fallback for selective_scan_cuda. "
    "Performance will be significantly slower than the CUDA implementation. "
    "Consider installing mamba-ssm with CUDA support for better performance.",
    UserWarning
)