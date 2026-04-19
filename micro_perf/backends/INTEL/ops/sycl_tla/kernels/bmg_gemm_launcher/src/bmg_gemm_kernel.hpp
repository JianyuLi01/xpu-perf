// SPDX-License-Identifier: Apache-2.0
//
// Templated BF16 x BF16 -> FP32 GEMM kernel for Intel BMG (Xe2),
// built on sycl-tla / cutlass collective builders.
//
// This header is the SINGLE place that knows how to materialise a
// `cutlass::gemm::device::GemmUniversalAdapter` for a given
// (TileShape, SubgroupShape, PipelineStages, SplitK) tuple.
//
// `bmg_gemm_instances.cpp` explicitly instantiates one `Run<...>` per
// configuration in the registry; the registry stores the resulting
// function pointers so the autotuner can call them by index.

#pragma once

#include "bmg_gemm_configs.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

// sycl-tla / cutlass headers (header-only; provided via include path).
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace bmg_gemm_launcher {

// Median of a small vector of latencies (microseconds).
inline double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// CPU reference GEMM (BF16xBF16 -> FP32) used only when verify=1.
// Slow on purpose; called once during autotune to gate a candidate.
void ReferenceGemm(const ProblemSize& p,
                   const std::vector<float>& A,
                   const std::vector<float>& B,
                   std::vector<float>& C);

bool VerifyClose(const std::vector<float>& got,
                 const std::vector<float>& ref,
                 double rtol, double atol);

// Templated launcher. One instantiation per candidate config.
//
// Template parameters
//   TileM, TileN, TileK     : CTA tile shape
//   SgM,   SgN,   SgK       : subgroup MMA shape
//   Stages                  : pipeline depth
//   SplitK                  : split-K factor (1 == disabled)
template <int TileM, int TileN, int TileK,
          int SgM,   int SgN,   int SgK,
          int Stages,
          int SplitK>
RunResult Run(const ProblemSize& problem, const RunOptions& opts) {
    using namespace cute;

    using ElementA      = cutlass::bfloat16_t;
    using ElementB      = cutlass::bfloat16_t;
    using ElementAcc    = float;
    using ElementC      = float;
    using ElementOutput = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using TileShape    = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using SubgroupTile = Shape<Int<SgM>,   Int<SgN>,   Int<SgK>>;

    // ArchTag selects the BMG (Xe2 / PVC successor) builder paths in sycl-tla.
    using ArchTag      = cutlass::arch::IntelXe;
    using OpClass      = cutlass::arch::OpClassTensorOp;

    // Mainloop: built by the collective builder; PipelineStages controls
    // software pipeline depth.
    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OpClass,
            ElementA, LayoutA, 1,
            ElementB, LayoutB, 1,
            ElementAcc,
            TileShape,
            SubgroupTile,
            cutlass::gemm::collective::StageCount<Stages>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OpClass,
            TileShape,
            SubgroupTile,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAcc, ElementAcc,
            ElementC, LayoutC, 1,
            ElementOutput, LayoutC, 1,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    const int M = problem.m;
    const int N = problem.n;
    const int K = problem.k;
    const int L = problem.l;

    // Allocate device buffers via SYCL USM. Use the default (in-order) queue.
    sycl::queue q{sycl::default_selector_v};

    auto alloc = [&](std::size_t bytes) {
        return sycl::malloc_device<std::byte>(bytes, q);
    };

    const std::size_t bytes_a = std::size_t(M) * K * L * sizeof(ElementA);
    const std::size_t bytes_b = std::size_t(K) * N * L * sizeof(ElementB);
    const std::size_t bytes_c = std::size_t(M) * N * L * sizeof(ElementC);

    auto* dA = reinterpret_cast<ElementA*>(alloc(bytes_a));
    auto* dB = reinterpret_cast<ElementB*>(alloc(bytes_b));
    auto* dC = reinterpret_cast<ElementC*>(alloc(bytes_c));
    auto* dD = reinterpret_cast<ElementOutput*>(alloc(bytes_c));

    if (!dA || !dB || !dC || !dD) {
        if (dA) sycl::free(dA, q);
        if (dB) sycl::free(dB, q);
        if (dC) sycl::free(dC, q);
        if (dD) sycl::free(dD, q);
        return RunResult{0.0, 0.0, false, false, "device allocation failed"};
    }

    // Initialise inputs with a deterministic but non-trivial pattern. The
    // host buffers are only kept around if verify=1.
    std::vector<float> hA, hB, hC, hRef;
    if (opts.verify) {
        hA.resize(std::size_t(M) * K);
        hB.resize(std::size_t(K) * N);
        hC.assign(std::size_t(M) * N, 0.0f);
        hRef.assign(std::size_t(M) * N, 0.0f);
        for (std::size_t i = 0; i < hA.size(); ++i) {
            hA[i] = static_cast<float>((i % 7) - 3) * 0.125f;
        }
        for (std::size_t i = 0; i < hB.size(); ++i) {
            hB[i] = static_cast<float>((i % 5) - 2) * 0.125f;
        }
        // Convert host floats -> bf16 on device via memcpy of cast values.
        std::vector<ElementA> hA_bf(hA.size());
        std::vector<ElementB> hB_bf(hB.size());
        for (std::size_t i = 0; i < hA.size(); ++i) hA_bf[i] = ElementA(hA[i]);
        for (std::size_t i = 0; i < hB.size(); ++i) hB_bf[i] = ElementB(hB[i]);
        q.memcpy(dA, hA_bf.data(), bytes_a).wait();
        q.memcpy(dB, hB_bf.data(), bytes_b).wait();
    } else {
        // Best-effort init: zero the buffers (kernel correctness is gated
        // by verify=1 paths only).
        q.memset(dA, 0, bytes_a).wait();
        q.memset(dB, 0, bytes_b).wait();
    }
    q.memset(dC, 0, bytes_c).wait();

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, L});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {dA, stride_A, dB, stride_B},
        {{1.0f, 0.0f}, dC, stride_C, dD, stride_D},
    };

    // Configure split-K: most sycl-tla universal-adapter paths expose a
    // hardware-info / scheduler arg structure carrying `splits`. We set it
    // when SplitK > 1; if the build of sycl-tla being used does not honour
    // it, the kernel will still run with split_k == 1 and the autotuner
    // will simply pick a different config.
    if constexpr (SplitK > 1) {
        args.scheduler.splits = SplitK;
    }

    Gemm gemm_op;

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        sycl::free(dA, q); sycl::free(dB, q);
        sycl::free(dC, q); sycl::free(dD, q);
        return RunResult{0.0, 0.0, false, false, "can_implement failed"};
    }

    const std::size_t ws_size = Gemm::get_workspace_size(args);
    void* d_workspace = nullptr;
    if (ws_size) {
        d_workspace = sycl::malloc_device<std::byte>(ws_size, q);
        if (!d_workspace) {
            sycl::free(dA, q); sycl::free(dB, q);
            sycl::free(dC, q); sycl::free(dD, q);
            return RunResult{0.0, 0.0, false, false, "workspace alloc failed"};
        }
    }

    status = gemm_op.initialize(args, d_workspace);
    if (status != cutlass::Status::kSuccess) {
        if (d_workspace) sycl::free(d_workspace, q);
        sycl::free(dA, q); sycl::free(dB, q);
        sycl::free(dC, q); sycl::free(dD, q);
        return RunResult{0.0, 0.0, false, false, "initialize failed"};
    }

    // Warmup
    for (int i = 0; i < opts.warmup; ++i) {
        gemm_op.run(q);
    }
    q.wait();

    // Measure: per-iteration device time via SYCL events; take median.
    std::vector<double> latencies_ms;
    latencies_ms.reserve(opts.iterations);
    for (int i = 0; i < opts.iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        gemm_op.run(q);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies_ms.push_back(ms);
    }
    const double med_ms = Median(latencies_ms);

    bool verified = true;
    if (opts.verify) {
        std::vector<ElementOutput> hOut(std::size_t(M) * N, 0.0f);
        q.memcpy(hOut.data(), dD, bytes_c).wait();
        std::vector<float> hOutF(hOut.begin(), hOut.end());
        ReferenceGemm(problem, hA, hB, hRef);
        verified = VerifyClose(hOutF, hRef, opts.rtol, opts.atol);
    }

    if (d_workspace) sycl::free(d_workspace, q);
    sycl::free(dA, q); sycl::free(dB, q);
    sycl::free(dC, q); sycl::free(dD, q);

    const double flop = 2.0 * double(M) * double(N) * double(K) * double(L);
    const double tflops = (med_ms > 0.0) ? (flop / (med_ms * 1e-3)) / 1e12 : 0.0;

    return RunResult{med_ms, tflops, verified, true, ""};
}

}  // namespace bmg_gemm_launcher
