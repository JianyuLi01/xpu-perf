import os
import sys
import json
import pathlib
import platform
import random
import shutil
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
try:
    from backends.INTEL.provider_intel import INTEL_PROVIDER
except Exception:
    INTEL_PROVIDER = {}
try:
    from backends.INTEL.env_intel import INTEL_ENV
except Exception:
    INTEL_ENV = {}


class BackendINTEL(Backend):
    def __init__(self):
        super().__init__()

    def get_backend_info(self):
        info_dict = {}

        # device相关
        info_dict["device_name"] = platform.processor() or platform.machine() or "Intel CPU"
        info_dict["device_count"] = 1
        info_dict["device_memory_mb"] = 0

        info_dict["torch_version"] = torch.__version__

        return info_dict

    def get_default_envs(self):
        return INTEL_ENV.get(self.backend_info.get("device_name", ""), {})

    def get_provider_info(self):
        return INTEL_PROVIDER

    def clean_extra_files(self):
        PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling")
        if PROFILER_DIR.exists():
            shutil.rmtree(PROFILER_DIR)


    """
    device management related
    """
    def get_torch_device_name(self):
        return "cpu"

    def get_device_name(self, index=0):
        return self.backend_info.get("device_name", "Intel CPU")

    def get_device_properties(self, index=0):
        return {
            "device_name": self.get_device_name(index),
        }

    def get_mem_info(self, index=0):
        try:
            import psutil
            mem = psutil.virtual_memory()
            return (mem.available, mem.total)
        except Exception:
            return (0, 0)

    def get_device_count(self):
        return 1, [0]

    def set_device(self, device_index: int):
        # CPU 不需要切换设备
        return

    def get_device(self):
        return 0

    def device_synchronize(self):
        # CPU 不需要同步
        return

    def empty_cache(self):
        # CPU 不需要清空缓存
        return


    """
    ccl related
    """
    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "gloo"


    def core_perf(
        self, op_instance,
        warmup_iterations, prefer_iterations,
        tensor_list,
        profiling=True
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        # CPU 上使用 wall-clock 计时, 不依赖 cuda event / profiler
        for i in range(warmup_iterations):
            index = random.randint(0, len(tensor_list) - 1)
            op_instance.core_run(tensor_list[index])

        self.device_synchronize()
        self.op_group_barrier(op_group=op_group, group_size=group_size)

        import time
        start_time = time.perf_counter()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        end_time = time.perf_counter()

        latency_us = (end_time - start_time) * 1e6 / prefer_iterations
        return latency_us, []
