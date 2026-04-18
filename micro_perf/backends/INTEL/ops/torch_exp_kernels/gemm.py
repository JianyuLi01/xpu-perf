import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[5])
)

from core.op import ProviderRegistry
from core.ops.tensor_gemm_ops import GemmOp

try:
    import torch_exp_kernels

    @ProviderRegistry.register_vendor_impl("gemm", "torch_exp_kernels")
    class TorchExpKernelsGemmOp(GemmOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_impl_run(self, tensor_mapping):
            a = tensor_mapping["a"]
            b = tensor_mapping["b"]
            c = torch_exp_kernels.gemm(a, b)
            return c

except Exception as e:
    print(f"[TorchExpKernelsGemmOp] Failed to register: {e}")
    pass
