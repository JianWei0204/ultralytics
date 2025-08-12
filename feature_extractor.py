from hook import Hook


class FeatureExtractor:
    """
    Helper class to extract features from specific layers in YOLOv8 model.
    """

    def __init__(self, model, layer_name='Adjust_Transformer'):
        self.model = model
        self.features = None
        self.hook = None
        self.layer_name = layer_name
        self.register_hook()

    def register_hook(self):
        """Register hook to capture output from the specified layer."""
        for name, module in self.model.named_modules():
            if self.layer_name in name or (hasattr(module, '__class__') and
                                           module.__class__.__name__ == self.layer_name):
                self.hook = Hook(module)
                break

        if self.hook is None:
            print(f"Warning: Could not find layer named {self.layer_name}")

    def get_features(self):
        """Get features from the last forward pass."""
        if self.hook is None:
            return None
        return self.hook.output

    def remove_hook(self):
        """Remove the hook when done."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None