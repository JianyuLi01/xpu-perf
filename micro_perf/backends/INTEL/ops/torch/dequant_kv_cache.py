import sys
import pathlib
from functools import partial

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import BasicOp
from core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype, get_attn_info
import torch


class DequantKVCacheOp(BasicOp):
    """Standalone INTEL dequant_kv_cache implementation.

    Upstream removed this op class; we keep it here for int8/float8
    KV-cache dequantization benchmarking on Intel GPUs.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm", "batch_llm"]:
            raise ValueError

        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if self.attn_mode not in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)

        # src (quant) dtype
        self.dtype = self.args_dict.get("dtype", "int8")
        if self.dtype not in ["int8", "float8"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dequant target dtype
        self.dst_dtype = self.args_dict.get("dst_dtype", "bfloat16")
        if self.dst_dtype not in ["bfloat16"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.quant_mode = self.args_dict.get("quant_mode", "static")
        if self.quant_mode not in ["static"]:
            raise ValueError

        # all tokens with same head/head_dim element pos share one scale
        if self.quant_mode == "static":
            self.quant_scale_shape = [self.kv_head_num, self.head_dim]

        self.input_tensor_info = {
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.kv_lens, dtype=dtype, device=device
                ),
            ),
            "k_scale": OpTensorInfo(
                shape=self.quant_scale_shape,
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
            "v_scale": OpTensorInfo(
                shape=self.quant_scale_shape,
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
        }
        self.output_tensor_info = {}

        if self.cache_type == "linear":
            self.input_tensor_info["slot_mapping"] = OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.slot_mapping, dtype=dtype, device=device
                ),
            )
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )

        elif self.cache_type == "paged":
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.batch_size, self.max_block_num_per_seq],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.block_table, dtype=dtype, device=device
                ),
            )
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )

        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum(
            [calc_tensor_size(info) for info in self.output_tensor_info.values()]
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = (
            calc_tensor_size(self.input_tensor_info["kv_lens"])
            + calc_tensor_size(self.input_tensor_info["k_scale"])
            + calc_tensor_size(self.input_tensor_info["v_scale"])
        )

        if self.cache_type == "linear":
            self.read_bytes += (
                calc_tensor_size(self.input_tensor_info["slot_mapping"])
                + calc_tensor_size(self.input_tensor_info["k_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
                + calc_tensor_size(self.input_tensor_info["v_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
            )
            self.write_bytes = (
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
                + calc_tensor_size(self.output_tensor_info["dequant_v_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
            )

        elif self.cache_type == "paged":
            self.read_bytes += (
                calc_tensor_size(self.input_tensor_info["block_table"])
                / self.batch_size
                / self.max_block_num_per_seq
                * self.num_kv_blocks
                + calc_tensor_size(self.input_tensor_info["k_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
                + calc_tensor_size(self.input_tensor_info["v_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
            )
            self.write_bytes = (
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
                + calc_tensor_size(self.output_tensor_info["dequant_v_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
            )

        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )

        self._run_func = self.dequant_kv_cache_run

    def dequant_kv_cache_run(self, tensor_mapping):
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        dequant_k_cache = tensor_mapping["dequant_k_cache"]
        dequant_v_cache = tensor_mapping["dequant_v_cache"]

        kv_lens = tensor_mapping["kv_lens"]
        k_scale = tensor_mapping["k_scale"]
        v_scale = tensor_mapping["v_scale"]

        if self.cache_type == "paged":
            block_table = tensor_mapping["block_table"]
            raise NotImplementedError(
                "DequantKVCacheOp paged cache not implemented yet."
            )

        if self.cache_type == "linear":
            slot_mapping = tensor_mapping["slot_mapping"]
            for batch_idx in range(self.batch_size):
                kv_slot_id = self.slot_mapping[batch_idx]
                kv_len = self.kv_lens[batch_idx]

                src_k_cache = k_cache[kv_slot_id, :, :kv_len, :].contiguous()
                src_v_cache = v_cache[kv_slot_id, :, :kv_len, :].contiguous()

                dst_k_cache = dequant_k_cache[kv_slot_id, :, :kv_len, :]
                dst_v_cache = dequant_v_cache[kv_slot_id, :, :kv_len, :]

                dst_k_cache.copy_(
                    torch.mul(
                        src_k_cache.to(k_scale.dtype), k_scale.unsqueeze(1)
                    ).to(dtype=self.dst_torch_dtype)
                )
                dst_v_cache.copy_(
                    torch.mul(
                        src_v_cache.to(v_scale.dtype), v_scale.unsqueeze(1)
                    ).to(dtype=self.dst_torch_dtype)
                )

        return dequant_k_cache, dequant_v_cache


OP_MAPPING = {"dequant_kv_cache": DequantKVCacheOp}
