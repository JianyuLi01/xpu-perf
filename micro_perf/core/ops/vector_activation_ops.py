import sys
import pathlib
import torch
from functools import partial

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



@register_base_impl
class SiluExtOp(BasicOp):
    """Extended SiLU operation: output = silu(src) * other.

    This fused operation applies the SiLU activation to ``src`` and
    element-wise multiplies the result with ``other``.  It is commonly
    used in gated MLP blocks of transformer-based LLMs (e.g. the gate
    projection path in SwiGLU).
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "other": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }

        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum(
            [calc_tensor_size(info) for info in self.output_tensor_info.values()]
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self.algo_size = 0
        self.bus_size = 0
        self.calc_flops = self.batch_size * self.dim_size

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        self._run_func = self.silu_ext_run

    def silu_ext_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        other = tensor_mapping["other"]
        dst = torch.nn.functional.silu(src) * other
        tensor_mapping["dst"] = dst
        return dst

