import importlib.metadata
import traceback

INTEL_PROVIDER = {}

try:
    INTEL_PROVIDER["intel_extension_for_pytorch"] = {
        "intel_extension_for_pytorch": importlib.metadata.version("intel_extension_for_pytorch")
    }
except:
    pass

try:
    INTEL_PROVIDER["sageattention"] = {
        "sageattention": importlib.metadata.version("sageattention")
    }
except:
    pass