import importlib.metadata

INTEL_PROVIDER = {}


# torch_exp_kernels: 实验性的 Intel 算子库
try:
    import torch_exp_kernels  # noqa: F401
    try:
        INTEL_PROVIDER["torch_exp_kernels"] = {
            "torch_exp_kernels": importlib.metadata.version("torch_exp_kernels"),
        }
    except importlib.metadata.PackageNotFoundError:
        INTEL_PROVIDER["torch_exp_kernels"] = {
            "torch_exp_kernels": "unknown",
        }
except Exception:
    pass
