import sys
import pathlib
from functools import partial
import torch
sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.op import BasicOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


class HeadRMSNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # pre-defined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.total_head_num = self.args_dict["total_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.norm_head_num = self.total_head_num  # TODO: can be different, use self.args_dict["norm_head_num"] later

        self.eps = 1e-5

        self.input_tensor_info = {
            "token_data": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "weight": OpTensorInfo(
                shape=[self.norm_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
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

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_head_num * self.norm_head_num + \
            calc_tensor_size(self.input_tensor_info["weight"])
        self.write_bytes = \
            calc_tensor_size(self.output_tensor_info["y"]) / self.total_head_num * self.norm_head_num
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
        self._run_func = self.head_rms_norm_run

    def head_rms_norm_run(self, tensor_mapping):
        # get pre-allocated input tensors
        token_data = tensor_mapping["token_data"]
        weight = tensor_mapping["weight"]

        # get pre-allocated output tensors
        y = tensor_mapping["y"]

        torch.ops.torch_ipex.head_rms_norm(weight, token_data, y, self.eps, self.norm_head_num)
        return y


class HeadRMSNormTorchOp(HeadRMSNormOp):
    def head_rms_norm_run(self, tensor_mapping):
        token_data = tensor_mapping["token_data"]
        weight = tensor_mapping["weight"]
        y = tensor_mapping["y"]

        head_data = token_data[:, :self.norm_head_num, :]
        normed = torch.nn.functional.rms_norm(
            head_data,
            normalized_shape=head_data.shape[-1:],
            weight=None,
            eps=self.eps
        )
        y[:, :self.norm_head_num, :] = normed * weight[:self.norm_head_num, :]
        return y


try:
    torch.ops.torch_ipex.head_rms_norm
    OP_MAPPING["head_rms_norm"] = HeadRMSNormOp
except Exception:
    pass
OP_MAPPING["head_rms_norm_torch"] = HeadRMSNormTorchOp
