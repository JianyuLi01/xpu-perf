"""sycl-tla BF16 GEMM provider.

This module no longer shells out to sycl-tla's ``examples/00_bmg_gemm``
binary. That example only ever instantiates a single hard-coded
``TileShape``/``TiledMma``/``PipelineStages`` configuration, with no
Split-K, and is documented as a teaching sample rather than a perf
kernel. Using it as the canonical sycl-tla GEMM number gives misleading
results for shapes that want a different tile (small-M decode, tall-K
split-K, large-square cooperative, ...).

Instead we invoke our own ``bmg_gemm_launcher`` (built from
``micro_perf/backends/INTEL/ops/sycl_tla/kernels/bmg_gemm_launcher``)
which:

* compiles a registry of ``TileShape x TiledMma x PipelineStages x SplitK``
  candidates,
* autotunes per ``(M, N, K)`` and reports the best,
* persists the choice in a JSONL cache so subsequent runs only pay for a
  single configuration.

The launcher emits JSON which is parsed here (no regex). If the launcher
binary is missing we keep the previous behaviour of NOT registering the
provider; we explicitly do **not** fall back to ``00_bmg_gemm``.
"""

import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[5])
)

from core.op import ProviderRegistry
from core.ops.tensor_gemm_ops import GemmOp


# --------------------------------------------------------------------------- #
# Launcher discovery                                                          #
# --------------------------------------------------------------------------- #

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_LAUNCHER_SRC_DIR = _THIS_DIR / "kernels" / "bmg_gemm_launcher"
_DEFAULT_LAUNCHER_BINARY = _LAUNCHER_SRC_DIR / "build" / "bin" / "bmg_gemm_launcher"

_DEFAULT_CACHE_PATH = (
    pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache")
    / "xpu-perf"
    / "bmg_gemm_autotune.jsonl"
)


def _resolve_launcher_binary():
    env = os.environ.get("BMG_GEMM_LAUNCHER", "").strip()
    if env:
        return pathlib.Path(env)
    return _DEFAULT_LAUNCHER_BINARY


_LAUNCHER_BINARY = _resolve_launcher_binary()


try:
    if not _LAUNCHER_BINARY.is_file():
        print(
            f"[WARNING] bmg_gemm_launcher binary not found at "
            f"{_LAUNCHER_BINARY}. sycl_tla_gemm provider will NOT be "
            f"available. Build it with:\n"
            f"  cmake -S {_LAUNCHER_SRC_DIR} -B {_LAUNCHER_SRC_DIR}/build "
            f"-DCMAKE_CXX_COMPILER=icpx -DSYCL_TLA_DIR=/path/to/sycl-tla\n"
            f"  cmake --build {_LAUNCHER_SRC_DIR}/build -j\n"
            f"or set BMG_GEMM_LAUNCHER to point at a prebuilt binary."
        )
        raise FileNotFoundError(str(_LAUNCHER_BINARY))


    @ProviderRegistry.register_vendor_impl("gemm", "sycl_tla_gemm")
    class SyclTlaGemmOp(GemmOp):
        """BF16 x BF16 -> FP32 GEMM backed by the autotuning launcher."""

        # Tracks what the launcher build was compiled to support. Update in
        # lockstep with the launcher's CLI / instances.
        SUPPORTED_DTYPES = ["bfloat16"]

        # Defaults; overridable via env so users can tune from outside.
        DEFAULT_ITERATIONS = int(os.environ.get("BMG_GEMM_ITERATIONS", "100"))
        DEFAULT_WARMUP = int(os.environ.get("BMG_GEMM_WARMUP", "3"))
        DEFAULT_VERIFY = int(os.environ.get("BMG_GEMM_VERIFY", "0"))
        TIMEOUT_SEC = int(os.environ.get("BMG_GEMM_TIMEOUT", "1800"))

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.dtype not in self.SUPPORTED_DTYPES:
                raise ValueError(
                    f"SyclTlaGemmOp only supports dtype in "
                    f"{self.SUPPORTED_DTYPES}, got {self.dtype}"
                )

            # Run launcher (autotune) and store parsed result.
            self._sycl_tla_result = self._run_launcher()

            # The benchmark harness's run/tensor allocation is a no-op;
            # the launcher binary did all of the device work and returned
            # the median latency we report from `summary()`.
            self._run_func = lambda tensor_mapping: None
            self._create_tensors_func = (
                lambda instance_num: [{}] * max(instance_num, 1)
            )

        # ------------------------------------------------------------------ #
        # Launcher invocation                                                #
        # ------------------------------------------------------------------ #

        def _build_command(self):
            cache_path = pathlib.Path(
                os.environ.get("BMG_GEMM_CACHE", str(_DEFAULT_CACHE_PATH))
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            return [
                str(_LAUNCHER_BINARY),
                "--m", str(self.M),
                "--n", str(self.N),
                "--k", str(self.K),
                "--l", "1",
                "--dtype", "bf16",
                "--autotune",
                "--iterations", str(self.DEFAULT_ITERATIONS),
                "--warmup", str(self.DEFAULT_WARMUP),
                "--verify", str(self.DEFAULT_VERIFY),
                "--cache", str(cache_path),
                "--json",
            ]

        def _run_launcher(self):
            cmd = self._build_command()
            env = os.environ.copy()
            # Match the flag the previous wrapper relied on; harmless to
            # the launcher itself but improves perf on real hardware.
            env.setdefault(
                "SYCL_PROGRAM_COMPILE_OPTIONS",
                "-ze-opt-large-register-file",
            )

            print(f"[SyclTlaGemmOp] Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.TIMEOUT_SEC,
                start_new_session=True,
                close_fds=True,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode != 0:
                raise RuntimeError(
                    f"bmg_gemm_launcher failed (rc={result.returncode}):\n"
                    f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                )

            # The launcher prints exactly one JSON object on stdout (with
            # `--json`). Be defensive: pick the last `{...}` block in case
            # informational lines were printed first.
            payload = self._extract_last_json_object(stdout)
            if payload is None:
                raise RuntimeError(
                    f"Failed to parse bmg_gemm_launcher JSON output:\n"
                    f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                )

            print(
                f"[SyclTlaGemmOp] best_config={payload.get('best_config')!r} "
                f"tflops={payload.get('tflops')} "
                f"latency_ms={payload.get('latency_ms')} "
                f"from_cache={payload.get('from_cache')} "
                f"verified={payload.get('verified')}"
            )

            tflops = float(payload.get("tflops", 0.0))
            latency_ms = float(payload.get("latency_ms", 0.0))
            if latency_ms <= 0.0:
                raise RuntimeError(
                    f"bmg_gemm_launcher returned non-positive latency: "
                    f"{payload}"
                )

            return {
                "tflops": tflops,
                "latency_ms": latency_ms,
                "best_config": payload.get("best_config"),
                "best_config_id": payload.get("best_config_id"),
                "from_cache": bool(payload.get("from_cache", False)),
                "verified": bool(payload.get("verified", False)),
                "all": payload.get("all", []),
            }

        @staticmethod
        def _extract_last_json_object(text):
            """Return the last balanced ``{...}`` JSON object in ``text``."""
            end = text.rfind("}")
            if end == -1:
                return None
            depth = 0
            start = -1
            for i in range(end, -1, -1):
                ch = text[i]
                if ch == "}":
                    depth += 1
                elif ch == "{":
                    depth -= 1
                    if depth == 0:
                        start = i
                        break
            if start == -1:
                return None
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None

        # ------------------------------------------------------------------ #
        # Reporting                                                          #
        # ------------------------------------------------------------------ #

        def summary(self, latency_us, kernel_mapping={}):
            if self._sycl_tla_result:
                latency_ms = self._sycl_tla_result.get("latency_ms", 0.0)
                if latency_ms > 0:
                    latency_us = latency_ms * 1000.0
            return super().summary(latency_us, kernel_mapping)


except Exception as e:
    print(f"[SyclTlaGemmOp] Failed to register: {e}")
    pass
