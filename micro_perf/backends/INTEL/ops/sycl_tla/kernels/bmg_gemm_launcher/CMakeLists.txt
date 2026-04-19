# SPDX-License-Identifier: Apache-2.0
#
# Stand-alone CMake project for the BMG (Xe2) GEMM autotuning launcher.
#
# This deliberately does NOT depend on sycl-tla's `examples/00_bmg_gemm`
# build target. We only consume sycl-tla's headers (it is effectively
# header-only).
#
# Required:
#   - icpx (Intel oneAPI DPC++ compiler) on PATH
#   - SYCL_TLA_DIR pointing at the sycl-tla repo root, OR a sibling
#     `sycl-tla` directory next to xpu-perf.
#
# Example:
#   cmake -S . -B build \
#         -DCMAKE_CXX_COMPILER=icpx \
#         -DSYCL_TLA_DIR=/path/to/sycl-tla
#   cmake --build build -j

cmake_minimum_required(VERSION 3.20)
project(bmg_gemm_launcher CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Locate sycl-tla (header-only consumption).
# ---------------------------------------------------------------------------
if(NOT DEFINED SYCL_TLA_DIR OR SYCL_TLA_DIR STREQUAL "")
    # Try sibling layout: <xpu-perf>/../sycl-tla
    get_filename_component(_xpu_perf_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../.." ABSOLUTE)
    if(EXISTS "${_xpu_perf_root}/../sycl-tla/include/cutlass/cutlass.h")
        set(SYCL_TLA_DIR "${_xpu_perf_root}/../sycl-tla" CACHE PATH "sycl-tla repo root")
    endif()
endif()

if(NOT SYCL_TLA_DIR OR NOT EXISTS "${SYCL_TLA_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
        "SYCL_TLA_DIR is not set or does not point to a sycl-tla checkout.\n"
        "Pass -DSYCL_TLA_DIR=/path/to/sycl-tla to cmake.")
endif()
message(STATUS "Using sycl-tla at: ${SYCL_TLA_DIR}")

# ---------------------------------------------------------------------------
# Compiler / SYCL flags.
# ---------------------------------------------------------------------------
add_compile_options(
    -fsycl
    -fsycl-targets=intel_gpu_bmg_g21
    -O3
    -ffast-math
)
add_link_options(
    -fsycl
    -fsycl-targets=intel_gpu_bmg_g21
)

# ---------------------------------------------------------------------------
# Target.
# ---------------------------------------------------------------------------
add_executable(bmg_gemm_launcher
    src/main.cpp
    src/bmg_gemm_instances.cpp
)

target_include_directories(bmg_gemm_launcher
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SYCL_TLA_DIR}/include
        ${SYCL_TLA_DIR}/tools/util/include
)

# Place the binary at build/bin/ so the Python wrapper's default lookup
# path works without configuration.
set_target_properties(bmg_gemm_launcher PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS bmg_gemm_launcher RUNTIME DESTINATION bin)
