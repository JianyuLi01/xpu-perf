import sys
import pathlib
import torch

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.tensor_gemm_ops import GemmOp

OP_MAPPING = {}

class INTELGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        if self.dtype == "float32":
            torch.set_float32_matmul_precision("highest")
        elif self.dtype == "tfloat32":
            torch.set_float32_matmul_precision("high")

    def __del__(self):
        torch.set_float32_matmul_precision("highest")
        getattr(super(), "__del__", lambda: None)()


OP_MAPPING["torch"] = INTELGemmOp
