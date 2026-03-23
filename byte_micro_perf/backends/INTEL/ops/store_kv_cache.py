import sys
import pathlib
from functools import partial
import torch
import random
sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import StoreKVCacheOp
from core.op import BasicOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


# generator for prefill mode
def generate_prefill_data(
    q_seq_len, cache_len
):
    q_lens = [q_seq_len]
    accum_q_lens = [0, q_seq_len]
    cache_lens = [cache_len]
    cache_slot_ids = [0]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


# generator for prefill_session_cache mode
def generate_prefill_session_cache_data(
        batch_size,
        target_q_len,
        aver_cache_len
):
    # random q_len, accum to target_q_len
    aver_q_len = target_q_len // batch_size
    q_len_remainder = target_q_len % batch_size
    q_len_offset = aver_q_len // 10
    q_lens = []
    for i in range(batch_size):
        q_lens.append(aver_q_len + (1 if i < q_len_remainder else 0))
    for i in range(batch_size):
        q_lens[i] += random.randint(-q_len_offset, q_len_offset)

    # accum q_lens
    accum_q_lens = [0]
    for i in range(batch_size):
        accum_q_lens.append(accum_q_lens[-1] + q_lens[i])

    # random cache_lens
    cache_lens = [aver_cache_len for _ in range(batch_size)]
    cache_offset = aver_cache_len // 10
    for i in range(batch_size):
        cache_lens[i] += random.randint(-cache_offset, cache_offset)

    # sequential cache_slot_ids
    cache_slot_ids = [i for i in range(batch_size)]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


# generator for decode mode
def generate_decode_data(
    batch_size,
    q_seq_len,
    aver_cache_len
):
    # fixed q_len
    q_lens = [q_seq_len for _ in range(batch_size)]

    # accum q_lens
    accum_q_lens = [0]
    for i in range(batch_size):
        accum_q_lens.append(accum_q_lens[-1] + q_lens[i])

    # random cache_lens
    cache_lens = [aver_cache_len for _ in range(batch_size)]
    cache_offset = aver_cache_len // 10
    for i in range(batch_size):
        cache_lens[i] += random.randint(-cache_offset, cache_offset)

    # sequential cache_slot_ids
    cache_slot_ids = [i for i in range(batch_size)]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


class StoreKVCacheIpexOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict.get("dst_dtype", "int8")
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.torch_dst_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

        self.mode = self.args_dict.get("mode", "prefill")
        if self.mode == "prefill":
            # [q_seq_len, total_head_num, head_dim]
            self.batch_size = 1
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_prefill_data(
                    self.q_seq_len,
                    self.cache_len
                )

        elif self.mode == "prefill_session_cache":
            # [accumed_num_tokens, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_prefill_session_cache_data(
                    self.batch_size,
                    self.q_seq_len,
                    self.cache_len
                )

        elif self.mode == "decode":
            # [batch_size * q_seq_len, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                generate_decode_data(
                    self.batch_size,
                    self.q_seq_len,
                    self.cache_len
                )

        else:
            raise NotImplementedError

        # accum q_lens
        self.num_tokens = sum(self.q_lens)
        # accum cache_lens
        self.num_cache_tokens = sum(self.cache_lens)
        # max q_len + cache_len
        self.max_kv_len = max(self.kv_lens)
        self.max_q_len = max(self.q_lens)

        self.input_tensor_info = {
            "packed_qkv": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "q_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ),
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ),
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
            ),
            "cache_slot_ids": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_slot_ids, dtype=dtype, device=device)
            ),
            "k_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dst_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "v_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dst_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            ),
            "k_scale": OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
            "v_scale": OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }

        self.output_tensor_info = {

        }

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) / self.total_head_num * 2 * self.kv_head_num + \
            calc_tensor_size(self.input_tensor_info["q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["accum_q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["cache_lens"]) + \
            calc_tensor_size(self.input_tensor_info["cache_slot_ids"]) + \
            calc_tensor_size(self.input_tensor_info["k_scale"]) + \
            calc_tensor_size(self.input_tensor_info["v_scale"])

        self.write_bytes = \
            calc_tensor_size(self.input_tensor_info["k_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens + \
            calc_tensor_size(self.input_tensor_info["v_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens

        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

        # run func
        self._run_func = self.store_kv_cache_run

    def store_kv_cache_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        q_lens = tensor_mapping["q_lens"]
        accum_q_lens = tensor_mapping["accum_q_lens"]
        cache_lens = tensor_mapping["cache_lens"]
        cache_slot_ids = tensor_mapping["cache_slot_ids"]
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]
        k_scale = tensor_mapping["k_scale"]
        v_scale = tensor_mapping["v_scale"]

        torch.ops.torch_ipex.store_kv_cache(
            packed_qkv, q_lens, accum_q_lens, cache_lens, cache_slot_ids, k_cache, v_cache, k_scale,
            v_scale, self.max_q_len)
        return k_cache, v_cache


try:
    torch.ops.torch_ipex.store_kv_cache
    OP_MAPPING["store_kv_cache"] = StoreKVCacheIpexOp
except Exception:
    pass


class StoreKVCacheTorchWrapper(StoreKVCacheOp):
    """Wrapper that maps INTEL workload keys (q_seq_len, mode, dst_dtype) to core-expected keys (q_len, attn_mode, cache_dtype)."""
    def __init__(self, args_dict, backend, *args, **kwargs):
        mapped_args = dict(args_dict)
        if "q_seq_len" in mapped_args and "q_len" not in mapped_args:
            mapped_args["q_len"] = mapped_args["q_seq_len"]
        if "mode" in mapped_args and "attn_mode" not in mapped_args:
            mapped_args["attn_mode"] = mapped_args["mode"]
        if "dst_dtype" in mapped_args and "cache_dtype" not in mapped_args:
            mapped_args["cache_dtype"] = mapped_args["dst_dtype"]
        super().__init__(mapped_args, backend, *args, **kwargs)


OP_MAPPING["store_kv_cache_torch"] = StoreKVCacheTorchWrapper
