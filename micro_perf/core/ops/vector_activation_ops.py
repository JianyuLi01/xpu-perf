import sys
import pathlib
import torch

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[2])
)

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp, register_base_impl
from core.ops.vector_sfu_ops import CosOp








@register_base_impl
class GeluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.gelu_run

    def gelu_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.gelu(src)
        tensor_mapping["dst"] = dst
        return dst



@register_base_impl
class SiluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.silu_run

    def silu_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.silu(src)
        tensor_mapping["dst"] = dst
        return dst

