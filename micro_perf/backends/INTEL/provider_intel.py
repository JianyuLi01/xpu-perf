import importlib.metadata
import traceback

INTEL_PROVIDER = {}


# https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    INTEL_PROVIDER["flash_attn_v2"] = {
        "flash_attn_v2": importlib.metadata.version("flash_attn")
    }
except:
    pass


# https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
    INTEL_PROVIDER["flash_attn_v3"] = {
        "flash_attn_v3": importlib.metadata.version("flash_attn"),
    }    
except:
    pass


# https://github.com/vllm-project/vllm-xpu-kernels
try:
    import vllm_xpu_kernels._C
    INTEL_PROVIDER["vllm_xpu_kernels"] = {
        "vllm_xpu_kernels": importlib.metadata.version("vllm-xpu-kernels"),
    }
except:
    pass


# torch_exp_kernels (only provides gemm)
try:
    import torch_exp_kernels
    INTEL_PROVIDER["torch_exp_kernels"] = {
        "torch_exp_kernels": importlib.metadata.version("torch_exp_kernels"),
    }
except:
    pass
