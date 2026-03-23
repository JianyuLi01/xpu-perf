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


# https://github.com/vllm-project/vllm
try:
    INTEL_PROVIDER["vllm"] = {
        "vllm": importlib.metadata.version("vllm"),
    }
except:
    pass


# https://github.com/flashinfer-ai/flashinfer
try:
    INTEL_PROVIDER["flashinfer"] = {
        "flashinfer": importlib.metadata.version("flashinfer-python"),
    }
except:
    pass

try:
    INTEL_PROVIDER["sageattention"] = {
        "sageattention": importlib.metadata.version("sageattention")
    }
except:
    pass
