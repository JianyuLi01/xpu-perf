import sys
import pathlib
from functools import partial
import torch
sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import ScaleDynamicQuantOp
from core.op import BasicOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


class ScaleDynamicQuantIpexOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name()
            )
        }

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

        # run func
        self._run_func = self.scale_dynamic_quant


    def scale_dynamic_quant(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]
        quant_out = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        torch.ops.torch_ipex.scale_dynamic_quant(hidden_states, smooth_scale,
                                                 quant_out, per_token_scale)
        return quant_out


try:
    torch.ops.torch_ipex.scale_dynamic_quant
    OP_MAPPING["scale_dynamic_quant"] = ScaleDynamicQuantIpexOp
except Exception:
    pass
OP_MAPPING["scale_dynamic_quant_torch"] = ScaleDynamicQuantOp
