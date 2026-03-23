import sys
import pathlib
import torch
import random
from functools import partial
from itertools import combinations
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp
sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import MoeSoftmaxTopkOp

class MoeSoftmaxTopkIpexOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # pre-defined attrs
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        self.compute_mode = self.args_dict["compute_mode"]
        if not self.compute_mode in ["pre-softmax", "post-softmax"]:
            raise NotImplementedError

        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        
        # input/output tensors
        self.input_tensor_info = {
            "gating_output": OpTensorInfo(
                shape=[self.num_tokens, self.num_experts], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "selected_experts": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
            ), 
            "moe_weights": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
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

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # safe softmax + topk
        self.calc_flops = self.num_tokens * (5 * self.num_experts + self.topk * self.num_experts)
        # reduce sum + divice
        if self.compute_mode == "pre-softmax":
            self.calc_flops += self.num_tokens * 2 * self.topk

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )

        # run func
        self._run_func = self.moe_softmax_topk_run


    def moe_softmax_topk_run(self, tensor_mapping):
        gating_output = tensor_mapping["gating_output"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]

        
        # softmax --> topk --> normlize
        if self.compute_mode == "pre-softmax":
            torch.ops.torch_ipex.moe_softmax_topk(gating_output, selected_experts, moe_weights, self.topk, True)
        # topk --> softmax
        elif self.compute_mode == "post-softmax":
            torch.ops.torch_ipex.moe_softmax_topk(gating_output, selected_experts, moe_weights, self.topk,False)
        else:
            raise NotImplementedError


try:
    torch.ops.torch_ipex.moe_softmax_topk
    OP_MAPPING = {"moe_softmax_topk": MoeSoftmaxTopkIpexOp}
except Exception:
    OP_MAPPING = {}
OP_MAPPING["moe_softmax_topk_torch"] = MoeSoftmaxTopkOp
