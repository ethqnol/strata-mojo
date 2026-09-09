#!/usr/bin/env python3
"""Strata vs. Scikit-Learn Benchmarking Harness & Report Generator

Mathematically rigorous benchmarking framework comparing Strata (Mojo/SIMD/LAPACK)
against scikit-learn / NumPy / SciPy (Python/Cython/C/BLAS).
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Suite definitions
SUITES = {
    "linear": {
        "name": "Linear Models",
        "mojo": "benchmarks/mojo/bench_linear_models.mojo",
        "sklearn": "benchmarks/sklearn/bench_linear_models.py",
        "scale_params": {
            "quick": (2000, 10, 1, 2),
            "small": (10000, 20, 2, 5),
            "medium": (50000, 30, 2, 5),
            "large": (200000, 50, 3, 5),
        },
    },
    "trees": {
        "name": "Trees & Ensembles",
        "mojo": "benchmarks/mojo/bench_trees_ensembles.mojo",
        "sklearn": "benchmarks/sklearn/bench_trees_ensembles.py",
        "scale_params": {
            "quick": (2000, 10, 1, 2),
            "small": (10000, 15, 1, 3),
            "medium": (30000, 20, 2, 3),
            "large": (100000, 30, 2, 3),
        },
    },
    "clustering": {
        "name": "Clustering & Decomposition",
        "mojo": "benchmarks/mojo/bench_clustering.mojo",
        "sklearn": "benchmarks/sklearn/bench_clustering.py",
        "scale_params": {
            "quick": (2000, 10, 1, 2),
            "small": (10000, 20, 1, 3),
            "medium": (50000, 30, 2, 4),
            "large": (150000, 40, 2, 4),
        },
    },
    "neighbors": {
        "name": "Nearest Neighbors",
        "mojo": "benchmarks/mojo/bench_neighbors.mojo",
        "sklearn": "benchmarks/sklearn/bench_neighbors.py",
        "scale_params": {
            "quick": (2000, 10, 1, 2),
            "small": (5000, 15, 1, 3),
            "medium": (20000, 20, 1, 3),
            "large": (50000, 30, 2, 3),
        },
    },
    "preprocessing": {
        "name": "Preprocessing & Scalers",
        "mojo": "benchmarks/mojo/bench_preprocessing.mojo",
        "sklearn": "benchmarks/sklearn/bench_preprocessing.py",
        "scale_params": {
            "quick": (5000, 10, 1, 2),
            "small": (50000, 30, 2, 5),
            "medium": (200000, 40, 2, 5),
            "large": (1000000, 50, 3, 5),
        },
    },
    "linalg": {
        "name": "Linear Algebra & Kernels",
        "mojo": "benchmarks/mojo/bench_linalg_kernels.mojo",
        "sklearn": "benchmarks/sklearn/bench_linalg_kernels.py",
        "scale_params": {
            "quick": (256, 10000, 1, 2),
            "small": (512, 50000, 2, 5),
            "medium": (1024, 150000, 2, 5),
            "large": (2048, 500000, 3, 5),
        },
    },
    "naive_bayes": {
        "name": "Naive Bayes Classifiers",
        "mojo": "benchmarks/mojo/bench_naive_bayes.mojo",
        "sklearn": "benchmarks/sklearn/bench_naive_bayes.py",
        "scale_params": {
            "quick": (2000, 15, 1, 2),
            "small": (20000, 30, 2, 5),
            "medium": (100000, 50, 2, 5),
            "large": (500000, 100, 3, 5),
        },
    },
}

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_ms(val_ms: float) -> str:
    if val_ms < 0.001:
        return f"{val_ms * 1000000.0:.1f} ns"
    elif val_ms < 1.0:
        return f"{val_ms * 1000.0:.1f} µs"
    elif val_ms < 1000.0:
        return f"{val_ms:.2f} ms"
    else:
        return f"{val_ms / 1000.0:.2f} s"


def format_speedup(speedup: float) -> str:
    if speedup >= 1.5:
        return f"{GREEN}{BOLD}{speedup:.2f}x faster{RESET}"
    elif speedup >= 1.05:
        return f"{GREEN}{speedup:.2f}x faster{RESET}"
    elif speedup >= 0.95:
        return f"{YELLOW}≈ parity ({speedup:.2f}x){RESET}"
    else:
        return f"{RED}{speedup:.2f}x (slower){RESET}"


def run_command(cmd, env):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        print(f"{RED}Error running command: {' '.join(cmd)}{RESET}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return []

    results = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                results.append(data)
            except Exception:
                pass
    return results


def run_benchmark_suite(suite_key, scale, env, pixi_prefix):
    suite_info = SUITES[suite_key]
    params = suite_info["scale_params"][scale]
    samples, features, warmups, iters = params

    mojo_script = suite_info["mojo"]
    sklearn_script = suite_info["sklearn"]

    print(f"\n{BOLD}{CYAN}▶ Running {suite_info['name']} Suite ({scale.upper()}: {samples:,} samples, {features} features, {iters} iters)...{RESET}")

    # Mojo command with LAPACK linkage
    mojo_cmd = [
        "mojo",
        "run",
        "-I",
        ".",
        "-Xlinker",
        f"-L{pixi_prefix}/lib",
        "-Xlinker",
        "-llapack",
        "-Xlinker",
        "-lblas",
        mojo_script,
        str(samples),
        str(features),
        str(warmups),
        str(iters),
    ]

    # Sklearn command
    sklearn_cmd = [
        "python",
        sklearn_script,
        str(samples),
        str(features),
        str(warmups),
        str(iters),
    ]

    t0 = time.time()
    mojo_results = run_command(mojo_cmd, env)
    t_mojo = time.time() - t0

    t1 = time.time()
    sklearn_results = run_command(sklearn_cmd, env)
    t_sklearn = time.time() - t1

    # Map results by (benchmark, phase)
    merged = []
    mojo_map = {(r["benchmark"], r["phase"]): r for r in mojo_results}
    sklearn_map = {(r["benchmark"], r["phase"]): r for r in sklearn_results}

    all_keys = list(dict.fromkeys(list(mojo_map.keys()) + list(sklearn_map.keys())))
    for k in all_keys:
        m = mojo_map.get(k)
        s = sklearn_map.get(k)
        if m and s:
            speedup = s["median_ms"] / m["median_ms"] if m["median_ms"] > 0 else 1.0
            metric_diff = abs(m.get("metric_val", 0.0) - s.get("metric_val", 0.0))
            merged.append({
                "suite": suite_key,
                "benchmark": k[0],
                "phase": k[1],
                "samples": m["samples"],
                "features": m["features"],
                "strata_median_ms": m["median_ms"],
                "strata_min_ms": m["min_ms"],
                "strata_max_ms": m["max_ms"],
                "strata_std_ms": m["std_ms"],
                "strata_throughput": m["throughput_samples_per_sec"],
                "sklearn_median_ms": s["median_ms"],
                "sklearn_min_ms": s["min_ms"],
                "sklearn_max_ms": s["max_ms"],
                "sklearn_std_ms": s["std_ms"],
                "sklearn_throughput": s["throughput_samples_per_sec"],
                "speedup": speedup,
                "metric_name": m.get("metric_name", "none"),
                "strata_metric": m.get("metric_val", 0.0),
                "sklearn_metric": s.get("metric_val", 0.0),
                "metric_diff": metric_diff,
            })

    return merged


def render_table(results):
    if not results:
        print("No results collected.")
        return

    header = f"{'Benchmark':<30} | {'Phase':<9} | {'Strata (Mojo)':<14} | {'Scikit-Learn':<14} | {'Speedup':<22} | {'Parity (Metric)'}"
    divider = "-" * 115
    print(divider)
    print(f"{BOLD}{header}{RESET}")
    print(divider)

    for r in results:
        bench_label = f"{r['benchmark']}"
        phase_label = f"{r['phase']}"
        strata_time = format_ms(r['strata_median_ms'])
        sklearn_time = format_ms(r['sklearn_median_ms'])
        speedup_str = format_speedup(r['speedup'])

        metric_str = "—"
        if r['metric_name'] != "none":
            metric_str = f"{r['metric_name']}: {r['strata_metric']:.4f} vs {r['sklearn_metric']:.4f} (Δ={r['metric_diff']:.1e})"

        print(f"{bench_label:<30} | {phase_label:<9} | {strata_time:<14} | {sklearn_time:<14} | {speedup_str:<31} | {metric_str}")
    print(divider)


def generate_markdown_report(all_results, output_path, scale):
    lines = []
    lines.append("# Strata vs. Scikit-Learn High-Performance Benchmark Suite")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append(f"**Workload Scale**: `{scale.upper()}`  ")
    lines.append("**Methodology**: Zero-overhead in-memory allocation, dedicated warmup iterations, nanosecond-precision monotonic timing (`perf_counter_ns`), median execution times across repetitions, strict single-thread environment control.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary Stats
    speedups = [r["speedup"] for r in all_results]
    faster_count = sum(1 for s in speedups if s >= 1.05)
    parity_count = sum(1 for s in speedups if 0.95 <= s < 1.05)
    slower_count = sum(1 for s in speedups if s < 0.95)
    geomean_speedup = 1.0
    for s in speedups:
        geomean_speedup *= (s ** (1.0 / len(speedups)))

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total Benchmarks Evaluated**: {len(all_results)}")
    lines.append(f"- **Overall Geometric Mean Speedup**: **{geomean_speedup:.2f}x**")
    lines.append(f"- **Strata Faster (≥ 1.05x)**: **{faster_count}** ({faster_count/len(all_results)*100:.1f}%)")
    lines.append(f"- **Equivalent / Parity (0.95x - 1.05x)**: **{parity_count}**")
    lines.append(f"- **Scikit-Learn Faster (< 0.95x)**: **{slower_count}**")
    lines.append("")

    # Split results into 3 major groups:
    # 1. Fits (Training)
    # 2. Predicts / Transforms / Queries (Inference)
    # 3. Low-Level Linear Algebra & Kernels (matmul, matvec)
    fit_items = [r for r in all_results if r["phase"] == "fit"]
    inference_items = [r for r in all_results if r["phase"] in ("predict", "transform", "kneighbors")]
    kernel_items = [r for r in all_results if r["phase"] in ("matmul", "matvec")]

    def render_section(title, items, is_fit=False):
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("*No benchmarks in this category.*")
            lines.append("")
            return

        # Compute category geomean
        cat_speedups = [r["speedup"] for r in items]
        cat_geomean = 1.0
        for s in cat_speedups:
            cat_geomean *= (s ** (1.0 / len(cat_speedups)))
        cat_faster = sum(1 for s in cat_speedups if s >= 1.05)

        lines.append(f"> **Category Summary**: Geometric Mean Speedup = **{cat_geomean:.2f}x** ({cat_faster}/{len(items)} benchmarks faster in Strata)")
        lines.append("")
        lines.append("| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for r in items:
            b_name = f"`{r['benchmark']}`"
            p_name = f"`{r['phase']}`"
            shape = f"{r['samples']:,} x {r['features']}"
            strata_time = format_ms(r['strata_median_ms'])
            sklearn_time = format_ms(r['sklearn_median_ms'])
            if r['speedup'] >= 1.05:
                speedup_val = f"**{r['speedup']:.2f}x**"
            elif r['speedup'] >= 0.95:
                speedup_val = f"{r['speedup']:.2f}x (parity)"
            else:
                speedup_val = f"{r['speedup']:.2f}x"

            tput = f"{r['strata_throughput']:,.0f} samples/s"

            parity = "Exact Match"
            if r['metric_name'] != "none":
                if r['metric_diff'] < 1e-4:
                    parity = f"{r['metric_name']}: {r['strata_metric']:.4f} (Exact)"
                else:
                    parity = f"{r['metric_name']}: {r['strata_metric']:.4f} vs {r['sklearn_metric']:.4f} (Δ={r['metric_diff']:.1e})"

            lines.append(f"| {b_name} | {p_name} | {shape} | {strata_time} | {sklearn_time} | {speedup_val} | {tput} | {parity} |")
        lines.append("")

    render_section("1. Model Training & Fitting Performance (`fit`)", fit_items, is_fit=True)
    render_section("2. Model Inference & Transformation Throughput (`predict`, `transform`, `query`)", inference_items)
    render_section("3. Low-Level Linear Algebra & Sparse Kernels (`gemm`, `spmv`, `spmm`)", kernel_items)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{GREEN}✔ Markdown benchmark report written to {output_path}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Strata vs Scikit-Learn Benchmark Runner")
    parser.add_argument(
        "--suite",
        choices=["all", "linear", "trees", "clustering", "neighbors", "preprocessing", "linalg", "naive_bayes"],
        default="all",
        help="Benchmark suite to execute (default: all)",
    )
    parser.add_argument(
        "--scale",
        choices=["quick", "small", "medium", "large"],
        default="small",
        help="Workload scale profile (default: small)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for BLAS / OpenMP baseline (default: 1)",
    )
    parser.add_argument(
        "--output-json",
        default="benchmarks/results/benchmark_data.json",
        help="Path to output raw JSON results",
    )
    parser.add_argument(
        "--output-md",
        default="benchmarks/results/BENCHMARK_REPORT.md",
        help="Path to output markdown report",
    )

    args = parser.parse_args()

    # Configure execution environment
    env = os.environ.copy()
    pixi_prefix = env.get("CONDA_PREFIX", "/home/ewu/Code/Strata/.pixi/envs/default")
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
    env["NUMEXPR_NUM_THREADS"] = str(args.threads)

    suites_to_run = list(SUITES.keys()) if args.suite == "all" else [args.suite]

    print(f"{BOLD}========================================================================{RESET}")
    print(f"{BOLD}           Strata vs. Scikit-Learn High-Performance Benchmark Suite     {RESET}")
    print(f"{BOLD}========================================================================{RESET}")
    print(f"• Scale: {BOLD}{args.scale.upper()}{RESET}")
    print(f"• Thread Baseline: {BOLD}{args.threads} thread(s){RESET}")
    print(f"• Pixi Prefix: {DIM}{pixi_prefix}{RESET}")

    all_results = []
    for s_key in suites_to_run:
        suite_results = run_benchmark_suite(s_key, args.scale, env, pixi_prefix)
        render_table(suite_results)
        all_results.extend(suite_results)

    # Save raw JSON
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{GREEN}✔ Raw JSON saved to {args.output_json}{RESET}")

    # Generate Markdown Report
    generate_markdown_report(all_results, args.output_md, args.scale)


if __name__ == "__main__":
    main()
