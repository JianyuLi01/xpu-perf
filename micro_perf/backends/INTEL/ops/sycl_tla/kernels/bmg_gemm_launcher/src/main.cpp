// SPDX-License-Identifier: Apache-2.0
//
// CLI entry point for `bmg_gemm_launcher`.
//
// Usage:
//   bmg_gemm_launcher --m M --n N --k K [--l 1] [--dtype bf16]
//                     [--autotune | --config-id N | --config-name NAME]
//                     [--iterations 10] [--warmup 3] [--verify 0|1]
//                     [--cache /path/to/cache.json] [--json] [--list]
//
// JSON output schema (when --json):
//   {
//     "m": M, "n": N, "k": K, "l": L, "dtype": "bf16",
//     "best_config": "Tile128x128x32_Sg8x16x16_S3_SK1",
//     "best_config_id": 2,
//     "tflops": 123.4,
//     "latency_ms": 0.123,
//     "verified": true,
//     "from_cache": false,
//     "all": [
//       {"config": "...", "config_id": 0, "latency_ms": .., "tflops": ..,
//        "verified": true, "ok": true, "error": ""},
//       ...
//     ]
//   }

#include "bmg_gemm_configs.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace bmg_gemm_launcher;

namespace {

struct CliArgs {
    int m = 0, n = 0, k = 0, l = 1;
    std::string dtype       = "bf16";
    bool        autotune    = false;
    bool        list_only   = false;
    int         config_id   = -1;
    std::string config_name;
    int         iterations  = 10;
    int         warmup      = 3;
    int         verify      = 0;
    std::string cache_path;
    bool        json        = false;
};

[[noreturn]] void Die(const std::string& msg) {
    std::fprintf(stderr, "[bmg_gemm_launcher] %s\n", msg.c_str());
    std::exit(2);
}

bool ParseInt(const char* s, int* out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!s || *s == 0 || (end && *end != 0)) return false;
    *out = static_cast<int>(v);
    return true;
}

CliArgs ParseCli(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) Die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if      (arg == "--m") { if (!ParseInt(need_val("--m"), &a.m)) Die("bad --m"); }
        else if (arg == "--n") { if (!ParseInt(need_val("--n"), &a.n)) Die("bad --n"); }
        else if (arg == "--k") { if (!ParseInt(need_val("--k"), &a.k)) Die("bad --k"); }
        else if (arg == "--l") { if (!ParseInt(need_val("--l"), &a.l)) Die("bad --l"); }
        else if (arg == "--dtype")       { a.dtype = need_val("--dtype"); }
        else if (arg == "--autotune")    { a.autotune = true; }
        else if (arg == "--list")        { a.list_only = true; }
        else if (arg == "--config-id")   { if (!ParseInt(need_val("--config-id"), &a.config_id)) Die("bad --config-id"); }
        else if (arg == "--config-name") { a.config_name = need_val("--config-name"); }
        else if (arg == "--iterations")  { if (!ParseInt(need_val("--iterations"), &a.iterations)) Die("bad --iterations"); }
        else if (arg == "--warmup")      { if (!ParseInt(need_val("--warmup"), &a.warmup)) Die("bad --warmup"); }
        else if (arg == "--verify")      { if (!ParseInt(need_val("--verify"), &a.verify)) Die("bad --verify"); }
        else if (arg == "--cache")       { a.cache_path = need_val("--cache"); }
        else if (arg == "--json")        { a.json = true; }
        else if (arg == "-h" || arg == "--help") {
            std::puts(
                "Usage: bmg_gemm_launcher --m M --n N --k K [--l 1] [--dtype bf16]\n"
                "                          [--autotune | --config-id N | --config-name NAME]\n"
                "                          [--iterations 10] [--warmup 3] [--verify 0|1]\n"
                "                          [--cache PATH] [--json] [--list]\n");
            std::exit(0);
        } else {
            Die("unknown argument: " + arg);
        }
    }
    if (!a.list_only) {
        if (a.m <= 0 || a.n <= 0 || a.k <= 0) Die("--m/--n/--k required and > 0");
        if (a.dtype != "bf16" && a.dtype != "bfloat16") {
            Die("only --dtype bf16 is supported in this build");
        }
        const bool has_id   = (a.config_id >= 0);
        const bool has_name = !a.config_name.empty();
        if (!a.autotune && !has_id && !has_name) {
            // Default to autotune when nothing was specified.
            a.autotune = true;
        }
    }
    return a;
}

// --- minimal JSON helpers (no external deps) ---------------------------------

std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// --- cache (tiny line-oriented JSONL: one record per line) -------------------
//
// We keep the on-disk format intentionally simple so we don't need a JSON
// parser in C++. Each line is: {"key":"...","cfg":"...","cfg_id":N,
// "tflops":..,"latency_ms":..}
// The `key` is M_N_K_L_dtype.

std::string CacheKey(const CliArgs& a) {
    std::ostringstream os;
    os << a.m << "_" << a.n << "_" << a.k << "_" << a.l << "_" << a.dtype;
    return os.str();
}

struct CacheHit {
    std::string cfg_name;
    int         cfg_id = -1;
    double      tflops = 0.0;
    double      latency_ms = 0.0;
};

std::optional<CacheHit> CacheLookup(const std::string& path,
                                    const std::string& key) {
    if (path.empty()) return std::nullopt;
    std::ifstream in(path);
    if (!in) return std::nullopt;
    const std::string needle = "\"key\":\"" + key + "\"";
    std::string line;
    while (std::getline(in, line)) {
        if (line.find(needle) == std::string::npos) continue;
        CacheHit h;
        // Crude field extraction; safe because writer only emits ASCII.
        auto get_str = [&](const std::string& field) -> std::string {
            const std::string p = "\"" + field + "\":\"";
            auto pos = line.find(p);
            if (pos == std::string::npos) return "";
            pos += p.size();
            auto end = line.find('"', pos);
            return line.substr(pos, end - pos);
        };
        auto get_num = [&](const std::string& field) -> double {
            const std::string p = "\"" + field + "\":";
            auto pos = line.find(p);
            if (pos == std::string::npos) return 0.0;
            pos += p.size();
            return std::strtod(line.c_str() + pos, nullptr);
        };
        h.cfg_name   = get_str("cfg");
        h.cfg_id     = static_cast<int>(get_num("cfg_id"));
        h.tflops     = get_num("tflops");
        h.latency_ms = get_num("latency_ms");
        return h;
    }
    return std::nullopt;
}

void EnsureParentDir(const std::string& path) {
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return;
    std::string dir = path.substr(0, pos);
    if (dir.empty()) return;
    // mkdir -p (only the immediate parent; deep paths should already exist).
    // It is fine for this to fail with EEXIST; any other failure will be
    // surfaced indirectly by CacheStore() failing to open the file.
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr,
            "[bmg_gemm_launcher] mkdir(%s) failed: %s\n",
            dir.c_str(), std::strerror(errno));
    }
}

void CacheStore(const std::string& path, const std::string& key,
                const std::string& cfg_name, int cfg_id,
                double tflops, double latency_ms) {
    if (path.empty()) return;
    EnsureParentDir(path);
    // Read all lines, drop existing entry with same key, append new one.
    std::vector<std::string> kept;
    {
        std::ifstream in(path);
        if (in) {
            const std::string needle = "\"key\":\"" + key + "\"";
            std::string line;
            while (std::getline(in, line)) {
                if (line.find(needle) == std::string::npos) {
                    kept.push_back(line);
                }
            }
        }
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "[bmg_gemm_launcher] cannot write cache %s\n",
                     path.c_str());
        return;
    }
    for (auto& l : kept) out << l << "\n";
    out << "{\"key\":\""   << JsonEscape(key)
        << "\",\"cfg\":\"" << JsonEscape(cfg_name)
        << "\",\"cfg_id\":" << cfg_id
        << ",\"tflops\":" << tflops
        << ",\"latency_ms\":" << latency_ms
        << "}\n";
}

// --- driver ------------------------------------------------------------------

struct ProfiledRun {
    int         cfg_id;
    std::string cfg_name;
    RunResult   res;
};

ProfiledRun RunSingle(int cfg_id, const ProblemSize& p, const RunOptions& opts) {
    const auto& cfgs = AllConfigs();
    const auto& cfg = cfgs.at(cfg_id);
    return ProfiledRun{cfg_id, cfg.name, cfg.run(p, opts)};
}

int FindByName(const std::string& name) {
    const auto& cfgs = AllConfigs();
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
        if (name == cfgs[i].name) return static_cast<int>(i);
    }
    return -1;
}

void EmitJson(const CliArgs& a,
              const ProfiledRun& best,
              const std::vector<ProfiledRun>& all,
              bool from_cache) {
    std::ostringstream os;
    os << "{";
    os << "\"m\":" << a.m << ",\"n\":" << a.n << ",\"k\":" << a.k
       << ",\"l\":" << a.l << ",\"dtype\":\"" << JsonEscape(a.dtype) << "\",";
    os << "\"best_config\":\"" << JsonEscape(best.cfg_name) << "\",";
    os << "\"best_config_id\":" << best.cfg_id << ",";
    os << "\"tflops\":" << best.res.tflops << ",";
    os << "\"latency_ms\":" << best.res.latency_ms << ",";
    os << "\"verified\":" << (best.res.verified ? "true" : "false") << ",";
    os << "\"from_cache\":" << (from_cache ? "true" : "false") << ",";
    os << "\"all\":[";
    for (std::size_t i = 0; i < all.size(); ++i) {
        const auto& r = all[i];
        if (i) os << ",";
        os << "{\"config\":\"" << JsonEscape(r.cfg_name) << "\","
           << "\"config_id\":" << r.cfg_id << ","
           << "\"latency_ms\":" << r.res.latency_ms << ","
           << "\"tflops\":" << r.res.tflops << ","
           << "\"verified\":" << (r.res.verified ? "true" : "false") << ","
           << "\"ok\":" << (r.res.ok ? "true" : "false") << ","
           << "\"error\":\"" << JsonEscape(r.res.error) << "\"}";
    }
    os << "]}";
    std::cout << os.str() << std::endl;
}

void EmitText(const ProfiledRun& best, bool from_cache) {
    // Keep a recognisable "Cutlass GEMM Performance" line so older log
    // post-processors keep working.
    std::printf("Cutlass GEMM Performance: [%.4f]TFlop/s (%.6f)ms\n",
                best.res.tflops, best.res.latency_ms);
    std::printf("[bmg_gemm_launcher] best=%s id=%d from_cache=%s verified=%s\n",
                best.cfg_name.c_str(), best.cfg_id,
                from_cache ? "true" : "false",
                best.res.verified ? "true" : "false");
}

}  // namespace

int main(int argc, char** argv) {
    CliArgs args = ParseCli(argc, argv);
    const auto& cfgs = AllConfigs();

    if (args.list_only) {
        for (std::size_t i = 0; i < cfgs.size(); ++i) {
            const auto& c = cfgs[i];
            std::printf("%3zu  %-44s  tile=%dx%dx%d sg=%dx%dx%d stages=%d splitK=%d\n",
                        i, c.name,
                        c.tile_shape.m, c.tile_shape.n, c.tile_shape.k,
                        c.subgroup_shape.m, c.subgroup_shape.n, c.subgroup_shape.k,
                        c.pipeline_stages, c.split_k);
        }
        return 0;
    }

    ProblemSize prob{args.m, args.n, args.k, args.l};
    RunOptions  opts;
    opts.iterations = args.iterations;
    opts.warmup     = args.warmup;
    opts.verify     = args.verify;

    // Cache fast path: if the user passed --cache and the (M,N,K,L,dtype)
    // is known, run only the cached config.
    if (args.autotune && !args.cache_path.empty()) {
        if (auto hit = CacheLookup(args.cache_path, CacheKey(args))) {
            int idx = hit->cfg_id;
            if (idx < 0 || idx >= static_cast<int>(cfgs.size()) ||
                std::string(cfgs[idx].name) != hit->cfg_name) {
                idx = FindByName(hit->cfg_name);
            }
            if (idx >= 0) {
                ProfiledRun pr = RunSingle(idx, prob, opts);
                if (pr.res.ok) {
                    if (args.json) EmitJson(args, pr, {pr}, /*from_cache=*/true);
                    else           EmitText(pr, /*from_cache=*/true);
                    return 0;
                }
            }
        }
    }

    // Single-config path.
    if (!args.autotune) {
        int idx = args.config_id;
        if (idx < 0 && !args.config_name.empty()) idx = FindByName(args.config_name);
        if (idx < 0 || idx >= static_cast<int>(cfgs.size())) {
            Die("invalid --config-id / --config-name");
        }
        ProfiledRun pr = RunSingle(idx, prob, opts);
        if (!pr.res.ok) {
            std::fprintf(stderr, "[bmg_gemm_launcher] config %s failed: %s\n",
                         pr.cfg_name.c_str(), pr.res.error.c_str());
            return 1;
        }
        if (args.json) EmitJson(args, pr, {pr}, false);
        else           EmitText(pr, false);
        return 0;
    }

    // Autotune path.
    std::vector<ProfiledRun> all;
    all.reserve(cfgs.size());

    // First pass: verify=1 (cheap, single-iteration) on each non-pruned
    // config to gate correctness, then re-time only verified ones.
    RunOptions verify_opts = opts;
    verify_opts.iterations = 1;
    verify_opts.warmup     = 0;
    verify_opts.verify     = 1;

    std::vector<int> survivors;
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
        if (ShouldPrune(cfgs[i], prob)) {
            ProfiledRun skipped{int(i), cfgs[i].name,
                RunResult{0.0, 0.0, false, false, "pruned"}};
            all.push_back(skipped);
            continue;
        }
        // Only do the expensive host-side verification when the user asked.
        if (args.verify) {
            ProfiledRun pr = RunSingle(int(i), prob, verify_opts);
            if (!pr.res.ok || !pr.res.verified) {
                all.push_back(pr);
                continue;
            }
        }
        survivors.push_back(int(i));
    }

    // Second pass: timed runs on survivors.
    ProfiledRun best{-1, "", RunResult{1e30, 0.0, false, false, "no candidate"}};
    for (int idx : survivors) {
        ProfiledRun pr = RunSingle(idx, prob, opts);
        all.push_back(pr);
        if (pr.res.ok && pr.res.latency_ms > 0.0 &&
            pr.res.latency_ms < best.res.latency_ms) {
            best = pr;
        }
    }

    if (best.cfg_id < 0) {
        std::fprintf(stderr,
            "[bmg_gemm_launcher] all candidates failed for shape "
            "M=%d N=%d K=%d L=%d\n", args.m, args.n, args.k, args.l);
        return 1;
    }

    if (!args.cache_path.empty()) {
        CacheStore(args.cache_path, CacheKey(args),
                   best.cfg_name, best.cfg_id,
                   best.res.tflops, best.res.latency_ms);
    }

    if (args.json) EmitJson(args, best, all, /*from_cache=*/false);
    else           EmitText(best, /*from_cache=*/false);
    return 0;
}
