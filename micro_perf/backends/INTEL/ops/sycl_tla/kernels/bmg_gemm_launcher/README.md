# bmg_gemm_launcher

A self-contained C++/SYCL launcher that benchmarks BF16 × BF16 → FP32 GEMM
on Intel **BMG (Xe2)** by **autotuning over a search space of**:

| Axis             | Candidates (initial) |
| ---------------- | -------------------- |
| `TileShape_MNK`  | `(128,128,32)`, `(128,256,32)`, `(256,128,32)`, `(64,128,32)`, `(128,64,32)`, `(32,128,32)`, `(128,32,32)`, `(64,64,32)`, `(128,128,64)` |
| `SubgroupShape`  | `(8,16,16)`, `(16,16,16)` |
| `PipelineStages` | `2`, `3`, `4` |
| `Split-K`        | `1`, `2`, `4` |

Configurations are explicitly instantiated in
[`src/bmg_gemm_instances.cpp`](src/bmg_gemm_instances.cpp).

> **Why this exists.** The `examples/00_bmg_gemm` binary that ships with
> sycl-tla compiles a *single* hard-coded configuration and exposes no
> way to change the tile / pipeline / split-K at runtime. That is fine
> as a teaching example, but it is not a usable performance kernel:
> small-M (decode), tall-K (split-K friendly), and large-square shapes
> all want very different configurations. The launcher in this directory
> picks the best one per shape.

## Building

Requirements:

* Intel oneAPI **DPC++** compiler (`icpx`) on `PATH`.
* A checkout of [sycl-tla](https://github.com/intel/sycl-tla) (header-only
  consumption — we do **not** depend on `examples/00_bmg_gemm`).

```bash
cd micro_perf/backends/INTEL/ops/sycl_tla/kernels/bmg_gemm_launcher

cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=icpx \
      -DSYCL_TLA_DIR=/path/to/sycl-tla
cmake --build build -j

# binary lands at build/bin/bmg_gemm_launcher
```

If `sycl-tla` lives next to `xpu-perf` (i.e. `<...>/sycl-tla` and
`<...>/xpu-perf` are siblings), `SYCL_TLA_DIR` is auto-detected.

The Python wrapper looks for the binary at:

1. `$BMG_GEMM_LAUNCHER` if set.
2. `<this dir>/build/bin/bmg_gemm_launcher`.

## CLI

```
bmg_gemm_launcher --m M --n N --k K [--l 1] [--dtype bf16]
                  [--autotune | --config-id N | --config-name NAME]
                  [--iterations 10] [--warmup 3] [--verify 0|1]
                  [--cache PATH] [--json] [--list]
```

* `--autotune` (default): try every (non-pruned) configuration and pick
  the one with the lowest median latency.
* `--config-id N` / `--config-name NAME`: run a single configuration.
  `--list` prints the registry (id, name, tile, subgroup, stages, splitK).
* `--cache PATH`: persistent JSONL cache keyed by `(M,N,K,L,dtype)`. On a
  cache hit the launcher only runs the cached configuration.
* `--verify 1`: also run a CPU reference and gate candidates on numerical
  correctness (slow, used only for the first autotune of each shape).
* `--json`: emit machine-readable JSON instead of the human-readable
  `Cutlass GEMM Performance: [TFlop/s] (ms)` line.

### JSON schema (`--json`)

```json
{
  "m": 4096, "n": 4096, "k": 4096, "l": 1, "dtype": "bf16",
  "best_config":    "Tile128x128x32_Sg8x16x16_S3_SK1",
  "best_config_id": 2,
  "tflops":     143.27,
  "latency_ms": 0.961,
  "verified":   true,
  "from_cache": false,
  "all": [
    {"config": "...", "config_id": 0, "latency_ms": 1.234,
     "tflops": 110.7, "verified": true, "ok": true, "error": ""}
  ]
}
```

## Adding a candidate

1. Add a `MAKE_CFG(...)` line in `src/bmg_gemm_instances.cpp`.
2. Rebuild.

Each new instantiation adds compile time, so prefer pruning to growth.
The pruning rules live in `ShouldPrune()` inside
[`include/bmg_gemm_configs.hpp`](include/bmg_gemm_configs.hpp).

## Cache

Default cache path used by the Python wrapper:
`~/.cache/xpu-perf/bmg_gemm_autotune.jsonl`. One line per shape; safe to
delete to force a re-tune.
