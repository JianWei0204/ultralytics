class Hook:
    """
    Hook to capture outputs from a PyTorch module.
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output

    def remove(self):
        self.hook.remove()