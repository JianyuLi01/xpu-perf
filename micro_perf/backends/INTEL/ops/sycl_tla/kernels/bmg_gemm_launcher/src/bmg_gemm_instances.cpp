// SPDX-License-Identifier: Apache-2.0
//
// Explicit instantiations of `Run<...>` for every candidate config, plus
// the registry returned by `AllConfigs()`.
//
// To add a new candidate:
//   1) Add a `MAKE_CFG(...)` line in `kAll` below.
//   2) Rebuild. (Each new instantiation adds compile time.)

#include "bmg_gemm_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace bmg_gemm_launcher {

void ReferenceGemm(const ProblemSize& p,
                   const std::vector<float>& A,
                   const std::vector<float>& B,
                   std::vector<float>& C) {
    // A: [M, K] row-major, B: [K, N] row-major, C: [M, N] row-major.
    const int M = p.m, N = p.n, K = p.k;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += double(A[std::size_t(m) * K + k]) *
                       double(B[std::size_t(k) * N + n]);
            }
            C[std::size_t(m) * N + n] = float(acc);
        }
    }
}

bool VerifyClose(const std::vector<float>& got,
                 const std::vector<float>& ref,
                 double rtol, double atol) {
    if (got.size() != ref.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(double(got[i]) - double(ref[i]));
        const double tol  = atol + rtol * std::fabs(double(ref[i]));
        if (diff > tol) {
            std::fprintf(stderr,
                "[verify] mismatch @%zu got=%g ref=%g diff=%g tol=%g\n",
                i, double(got[i]), double(ref[i]), diff, tol);
            return false;
        }
    }
    return true;
}

namespace {

#define MAKE_CFG(NAME, TM, TN, TK, SGM, SGN, SGK, ST, SK)            \
    GemmInstance{                                                    \
        NAME,                                                        \
        GemmShape3{TM, TN, TK},                                      \
        GemmShape3{SGM, SGN, SGK},                                   \
        ST,                                                          \
        SK,                                                          \
        &Run<TM, TN, TK, SGM, SGN, SGK, ST, SK>                      \
    }

// Curated initial search space. Each config name encodes the template
// parameters so the autotuner output is human-readable.
const std::vector<GemmInstance> kAll = {
    // ---- balanced square tiles ----
    MAKE_CFG("Tile128x128x32_Sg8x16x16_S2_SK1",  128, 128, 32, 8, 16, 16, 2, 1),
    MAKE_CFG("Tile128x128x32_Sg8x16x16_S3_SK1",  128, 128, 32, 8, 16, 16, 3, 1),
    MAKE_CFG("Tile128x128x32_Sg16x16x16_S3_SK1", 128, 128, 32, 16, 16, 16, 3, 1),

    // ---- M-heavy / N-heavy ----
    MAKE_CFG("Tile256x128x32_Sg8x16x16_S3_SK1",  256, 128, 32, 8, 16, 16, 3, 1),
    MAKE_CFG("Tile128x256x32_Sg8x16x16_S3_SK1",  128, 256, 32, 8, 16, 16, 3, 1),

    // ---- skinny tiles for small M (decode) ----
    MAKE_CFG("Tile32x128x32_Sg8x16x16_S2_SK1",   32, 128, 32, 8, 16, 16, 2, 1),
    MAKE_CFG("Tile64x128x32_Sg8x16x16_S3_SK1",   64, 128, 32, 8, 16, 16, 3, 1),
    MAKE_CFG("Tile128x32x32_Sg8x16x16_S2_SK1",   128, 32, 32, 8, 16, 16, 2, 1),
    MAKE_CFG("Tile128x64x32_Sg8x16x16_S3_SK1",   128, 64, 32, 8, 16, 16, 3, 1),

    // ---- small tile, deeper pipeline ----
    MAKE_CFG("Tile64x64x32_Sg8x16x16_S4_SK1",    64, 64, 32, 8, 16, 16, 4, 1),

    // ---- large-K: TileK=64 ----
    MAKE_CFG("Tile128x128x64_Sg8x16x16_S2_SK1",  128, 128, 64, 8, 16, 16, 2, 1),
    MAKE_CFG("Tile128x128x64_Sg16x16x16_S2_SK1", 128, 128, 64, 16, 16, 16, 2, 1),

    // ---- Split-K variants for tall/skinny (M*N small, K huge) ----
    MAKE_CFG("Tile64x64x32_Sg8x16x16_S3_SK2",    64, 64, 32, 8, 16, 16, 3, 2),
    MAKE_CFG("Tile64x64x32_Sg8x16x16_S3_SK4",    64, 64, 32, 8, 16, 16, 3, 4),
    MAKE_CFG("Tile128x128x32_Sg8x16x16_S3_SK2",  128, 128, 32, 8, 16, 16, 3, 2),
};

#undef MAKE_CFG

}  // namespace

const std::vector<GemmInstance>& AllConfigs() { return kAll; }

}  // namespace bmg_gemm_launcher
