import sys
import pathlib
sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.llm_ops import MoeGatingGemmOp

OP_MAPPING = {}


class MoeGatingGemmTorchOp(MoeGatingGemmOp):
    """Override vendor_parser to accept bfloat16/float16 input with float32 output."""

    def vendor_parser(self):
        self.dst_dtype = self.args_dict.get("dst_dtype", "float32")

        if self.dtype in ("float16", "bfloat16", "float32") and self.dst_dtype in ("float32", "float16"):
            pass
        else:
            raise ValueError(
                f"MoeGatingGemmTorchOp supports float16/bfloat16/float32-->float32/float16, "
                f"but got {self.dtype}-->{self.dst_dtype}"
            )


OP_MAPPING["torch"] = MoeGatingGemmTorchOp
