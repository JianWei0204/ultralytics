# YOLOv8 Model with Domain Adaptation Support
# Wrapper to extract intermediate features for domain adaptation

import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = "create_domain_adapted_model"


class YOLOv8WithFeatureExtraction(nn.Module):
    """
    Wrapper around YOLOv8 model to extract intermediate features.

    This class modifies the forward pass to enable feature extraction
    from the P4 layer (after backbone, before detection head).
    """

    def __init__(self, base_model):
        """
        Initialize with base YOLOv8 model.

        Args:
            base_model: The original YOLOv8 DetectionModel instance
        """
        super().__init__()
        self.base_model = base_model

        # Copy attributes from base model
        for attr in ['nc', 'names', 'args', 'stride', 'yaml']:
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

        # Flag to control feature extraction
        self.extract_features = False
        self.extracted_features = {}

        # Register hooks to extract features
        self._register_feature_hooks()

    def _register_feature_hooks(self):
        """Register forward hooks to extract features from P4 layer."""

        def hook_fn(name):
            def hook(module, input, output):
                if self.extract_features:
                    self.extracted_features[name] = output

            return hook

        # Find and register hook for P4 layer
        # This needs to be adapted based on actual YOLOv8 architecture
        if hasattr(self.base_model, 'model') and isinstance(self.base_model.model, nn.Sequential):
            model_layers = self.base_model.model

            # In YOLOv8, P4 is typically after the backbone
            # Look for the layer that corresponds to P4 output
            for i, layer in enumerate(model_layers):
                # Register hook on the layer that produces P4 features
                # This is architecture-specific and may need adjustment
                if i == 6:  # Assuming P4 is at index 6 (needs verification)
                    layer.register_forward_hook(hook_fn(f'P4_layer_{i}'))
                    break

    def forward(self, x, extract_features=False):
        """
        Forward pass with optional feature extraction.

        Args:
            x: Input batch (for training) or image tensor (for inference)
            extract_features: Whether to extract intermediate features

        Returns:
            If training (x is dict): loss and loss_items like original model
            If inference (x is tensor): predictions like original model
            If extract_features=True: also returns extracted features
        """
        self.extract_features = extract_features
        self.extracted_features = {}

        # Call original model forward
        result = self.base_model(x)

        if extract_features:
            return result, self.extracted_features
        else:
            return result

    def extract_p4_features(self, x):
        """
        Extract P4 layer features specifically.

        Args:
            x: Input image tensor

        Returns:
            P4 features tensor
        """
        self.extract_features = True
        self.extracted_features = {}

        # Run forward pass
        with torch.no_grad():
            _ = self.base_model(x)

        # Return P4 features
        for key, features in self.extracted_features.items():
            if 'P4' in key:
                return features

        # Fallback: return first extracted feature if P4 not found
        if self.extracted_features:
            return next(iter(self.extracted_features.values()))

        # If no features extracted, implement direct feature extraction
        return self._direct_p4_extraction(x)

    def _direct_p4_extraction(self, x):
        """
        Direct P4 feature extraction by running partial forward pass.

        This method navigates through the model layers to extract P4 features.
        """
        if not hasattr(self.base_model, 'model'):
            raise NotImplementedError("Cannot extract P4 features: model structure not recognized")

        # Navigate through model layers to P4
        current = x
        model_layers = self.base_model.model

        # Run through backbone layers until P4
        # This is a simplified approach and may need adjustment
        try:
            for i, layer in enumerate(model_layers):
                current = layer(current)

                # Stop at layer that produces P4 features
                # Adjust this condition based on actual model architecture
                if i == 6:  # Assuming P4 is at index 6
                    return current

        except Exception as e:
            raise RuntimeError(f"Failed to extract P4 features: {e}")

        return current

    def enable_feature_extraction(self):
        """Enable feature extraction mode."""
        self.extract_features = True

    def disable_feature_extraction(self):
        """Disable feature extraction mode."""
        self.extract_features = False
        self.extracted_features = {}

    def get_extracted_features(self):
        """Get currently extracted features."""
        return self.extracted_features.copy()

    def load(self, weights):
        """Load weights (delegate to base model)."""
        return self.base_model.load(weights)

    def fuse(self):
        """Fuse model layers (delegate to base model)."""
        if hasattr(self.base_model, 'fuse'):
            return self.base_model.fuse()

    def info(self, verbose=False, imgsz=640):
        """Get model info (delegate to base model)."""
        if hasattr(self.base_model, 'info'):
            return self.base_model.info(verbose, imgsz)

    def __getattr__(self, name):
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def create_domain_adapted_model(base_model):
    """
    Create a domain-adapted version of YOLOv8 model.

    Args:
        base_model: Original YOLOv8 DetectionModel

    Returns:
        YOLOv8WithFeatureExtraction wrapper
    """
    return YOLOv8WithFeatureExtraction(base_model)


class P4FeatureExtractor(nn.Module):
    """
    Standalone P4 feature extractor for cases where model modification is not possible.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture P4 features."""

        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.clone()

            return hook

        # Register hooks on relevant layers
        if hasattr(self.model, 'model'):
            for i, layer in enumerate(self.model.model):
                if i == 6:  # Adjust based on actual P4 layer index
                    handle = layer.register_forward_hook(hook_fn(f'P4_{i}'))
                    self.hooks.append(handle)

    def extract(self, x):
        """Extract P4 features from input."""
        self.features = {}

        with torch.no_grad():
            _ = self.model(x)

        # Return P4 features
        for key, features in self.features.items():
            if 'P4' in key:
                return features

        return None

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []