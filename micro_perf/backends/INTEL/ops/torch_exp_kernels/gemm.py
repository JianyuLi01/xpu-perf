import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.tensor_gemm_ops import GemmOp

OP_MAPPING = {}

try:
    import torch_exp_kernels

    class TorchExpKernelsGemmOp(GemmOp):
        def vendor_impl_run(self, tensor_mapping):
            a = tensor_mapping["a"]
            b = tensor_mapping["b"]
            c = torch_exp_kernels.gemm(a, b)
            return c

    OP_MAPPING["torch_exp_kernels"] = TorchExpKernelsGemmOp
except ImportError:
    pass
