// SPDX-License-Identifier: Apache-2.0
//
// Configuration registry for the BMG (Xe2) GEMM launcher.
//
// Each entry in kAllConfigs is a candidate that the autotuner will try.
// The actual kernel template (BmgGemmRunner) is parameterised on:
//
//   TileShape_MNK   : CTA tile (M, N, K)
//   SubgroupShape   : per-subgroup MMA tile (M, N, K)
//   PipelineStages  : software pipeline depth (mainloop stages)
//   SplitK          : number of split-K partitions (1 == no split-K)
//
// Adding a new candidate is a one-line change in this file plus the matching
// explicit instantiation in `bmg_gemm_instances.cpp`.

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace bmg_gemm_launcher {

struct GemmShape3 {
    int m;
    int n;
    int k;
};

// Result returned by a single profiling run.
struct RunResult {
    double latency_ms;     // median over `iterations` runs (device time)
    double tflops;         // 2*M*N*K / latency
    bool   verified;       // true if numerical verification passed (or skipped)
    bool   ok;             // false => kernel was not launched (pruned / failed)
    std::string error;     // populated when ok == false
};

struct ProblemSize {
    int m;
    int n;
    int k;
    int l = 1;             // batch
};

struct RunOptions {
    int iterations  = 10;  // measured iterations
    int warmup      = 3;
    int verify      = 0;   // 0/1
    double rtol     = 1e-2;
    double atol     = 1e-2;
};

// Forward declaration of the polymorphic instance interface.
struct GemmInstance {
    const char* name;                  // human readable: "Tile128x128x32_Sg8x16x16_S3_SK1"
    GemmShape3  tile_shape;            // CTA tile MNK
    GemmShape3  subgroup_shape;        // subgroup MMA shape
    int         pipeline_stages;
    int         split_k;
    // Function pointer to the templated launcher for this configuration.
    RunResult (*run)(const ProblemSize&, const RunOptions&);
};

// Filled by `bmg_gemm_instances.cpp`.
const std::vector<GemmInstance>& AllConfigs();

// Returns true when the configuration cannot possibly be efficient for the
// given shape (used to prune the search space before launching).
inline bool ShouldPrune(const GemmInstance& cfg, const ProblemSize& p) {
    // 1) Drop configurations where the CTA tile is larger than the problem
    //    dimension by a wide margin: M < TileM/2 means most CTA work is
    //    masked off and we should pick a smaller tile instead. The 32
    //    floor leaves the smallest tiles in the search space alone --
    //    they are the ones that handle very small M/N best.
    if (p.m * 2 < cfg.tile_shape.m && cfg.tile_shape.m > 32) return true;
    if (p.n * 2 < cfg.tile_shape.n && cfg.tile_shape.n > 32) return true;

    // 2) Split-K only makes sense when there is enough K to amortise the
    //    extra reduction AND when (M*N) tiles aren't already enough to
    //    fill the device. Thresholds:
    //      K  >= 2048    : minimum K for the reduction overhead to pay off.
    //      M*N <= 1M     : above this, M*N tiles already saturate the GPU.
    if (cfg.split_k > 1) {
        if (p.k < 2048) return true;
        if (static_cast<long long>(p.m) * p.n > 1024LL * 1024LL) return true;
        // Also need K to be divisible into split_k*TileK chunks reasonably.
        if (p.k < cfg.split_k * cfg.tile_shape.k * 2) return true;
    }

    // 3) Pipeline stages must fit; very deep pipelines on tiny K waste smem.
    if (cfg.pipeline_stages > 2 && p.k < cfg.tile_shape.k * cfg.pipeline_stages) {
        return true;
    }

    return false;
}

}  // namespace bmg_gemm_launcher
