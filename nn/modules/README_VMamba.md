# VMamba Fallback Implementation

This implementation provides fallback support for VMamba (Vision Mamba) operations when CUDA extensions are not available.

## Problem Solved

The original VMamba implementation depends on optimized CUDA extensions:
- `selective_scan_cuda` from mamba_ssm
- `causal_conv1d_cuda` from causal-conv1d

When these extensions fail to compile or are not available, the code crashes with import errors. This fallback implementation solves this by providing pure PyTorch alternatives.

## Features

- **Graceful Fallback**: Automatically detects missing CUDA extensions and falls back to PyTorch implementations
- **API Compatibility**: Maintains the same interface as the original implementations
- **Informative Warnings**: Provides clear warnings about performance implications
- **Full Functionality**: All operations work correctly, though with reduced performance

## Files Added

1. **`selective_scan_fallback.py`**: Pure PyTorch implementation of selective scan operations
2. **`causal_conv1d_fallback.py`**: Pure PyTorch implementation of causal 1D convolution
3. **`vmamba.py`**: VMamba implementation with automatic fallback support

## Usage

### Basic Usage

```python
# Import will work regardless of CUDA extension availability
from ultralytics.nn.modules import VMamba, VMambaBlock, Mamba

# Create a VMamba model
model = VMamba(d_model=256, n_layers=6)

# Use it like any PyTorch model
x = torch.randn(batch, seq_len, d_model)
output = model(x)
```

### With Warnings

When CUDA extensions are not available, you'll see informative warnings:

```
UserWarning: mamba_ssm not available, using PyTorch fallback implementation. 
Install mamba_ssm for better performance: pip install mamba-ssm

UserWarning: causal_conv1d not available, using PyTorch fallback implementation. 
Install causal-conv1d for better performance: pip install causal-conv1d
```

### Direct Component Usage

```python
# Use individual components
from ultralytics.nn.modules.selective_scan_fallback import selective_scan_fn
from ultralytics.nn.modules.causal_conv1d_fallback import causal_conv1d_fn, CausalConv1dFallback

# Selective scan operation
result = selective_scan_fn(u, delta, A, B, C, D)

# Causal convolution
conv = CausalConv1dFallback(in_channels=64, out_channels=128, kernel_size=3)
output = conv(x)
```

## Expected Behavior

### With CUDA Extensions Available
- Uses optimized CUDA implementations
- Maximum performance
- No warnings

### Without CUDA Extensions (Fallback Mode)
- Uses pure PyTorch implementations
- Reduced performance but full functionality
- Informative warnings about performance implications
- Same API and behavior

## Performance Notes

- **CUDA Extensions**: Optimized for maximum performance
- **Fallback Mode**: Slower but fully functional
- **Memory Usage**: Fallback may use more memory due to unoptimized operations
- **Gradient Computation**: All operations are differentiable in both modes

## Testing

The implementation has been tested to ensure:
- Correct output shapes
- Numerical consistency
- API compatibility
- Proper error handling
- Warning behavior

## Installation Requirements

### Minimum (Fallback Mode)
```bash
pip install torch
```

### Optimal (Full Performance)
```bash
pip install torch
pip install mamba-ssm
pip install causal-conv1d
```

## Compatibility

- Works with any PyTorch version that supports the required operations
- Compatible with CPU and GPU execution
- Maintains backward compatibility with existing VMamba code

## Examples

See the demo script for comprehensive examples of all functionality.