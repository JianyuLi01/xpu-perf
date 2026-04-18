import importlib.metadata

INTEL_PROVIDER = {}


try:
    import torch_exp_kernels

    INTEL_PROVIDER["torch_exp_kernels"] = {
        "torch_exp_kernels": importlib.metadata.version("torch_exp_kernels"),
    }
except ImportError:
    pass
