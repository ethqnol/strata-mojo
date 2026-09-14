from std.time import perf_counter_ns
from std.sys import argv
from strata import (
    Matrix,
    CSRMatrix,
    gemm,
    dense_dot_vec,
    spmv,
    spmm,
    spgemm,
)
from benchmarks.mojo.bench_utils import (
    BenchTimer,
    make_synthetic_sparse,
)
from strata.utils.random import PRNG


def run_gemm(size: Int, warmups: Int, iters: Int) raises:
    var A = Matrix[DType.float64](size, size, 1.5)
    var B = Matrix[DType.float64](size, size, 2.5)

    for _ in range(warmups):
        _ = gemm(A, B)

    var timer = BenchTimer()

    for _ in range(iters):
        var t0 = perf_counter_ns()
        _ = gemm(A, B)
        var t1 = perf_counter_ns()
        timer.add(t1 - t0)

    var res = timer.compute_stats("Dense_GEMM", "matmul", size, size)
    print(res.to_json())


def run_dense_dot_vec(rows: Int, cols: Int, warmups: Int, iters: Int) raises:
    var A = Matrix[DType.float64](rows, cols, 1.25)
    var x = List[Scalar[DType.float64]](capacity=cols)
    for _ in range(cols):
        x.append(0.75)

    for _ in range(warmups):
        _ = dense_dot_vec(A, x)

    var timer = BenchTimer()

    for _ in range(iters):
        var t0 = perf_counter_ns()
        _ = dense_dot_vec(A, x)
        var t1 = perf_counter_ns()
        timer.add(t1 - t0)

    var res = timer.compute_stats("Dense_Dot_Vec", "matvec", rows, cols)
    print(res.to_json())


def run_spmv(dim: Int, nnz_per_row: Int, warmups: Int, iters: Int) raises:
    var A = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=42)
    var x = List[Scalar[DType.float64]](capacity=dim)
    for _ in range(dim):
        x.append(1.0)

    for _ in range(warmups):
        _ = spmv(A, x)

    var timer = BenchTimer()

    for _ in range(iters):
        var t0 = perf_counter_ns()
        _ = spmv(A, x)
        var t1 = perf_counter_ns()
        timer.add(t1 - t0)

    var res = timer.compute_stats("Sparse_SpMV", "matvec", dim, dim)
    print(res.to_json())


def run_spmm(
    dim: Int, k_cols: Int, nnz_per_row: Int, warmups: Int, iters: Int
) raises:
    var A = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=42)
    var B = Matrix[DType.float64](dim, k_cols, 1.0)

    for _ in range(warmups):
        _ = spmm(A, B)

    var timer = BenchTimer()

    for _ in range(iters):
        var t0 = perf_counter_ns()
        _ = spmm(A, B)
        var t1 = perf_counter_ns()
        timer.add(t1 - t0)

    var res = timer.compute_stats("Sparse_SpMM", "matmul", dim, k_cols)
    print(res.to_json())


def run_spgemm(dim: Int, nnz_per_row: Int, warmups: Int, iters: Int) raises:
    var A = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=42)
    var B = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=43)

    for _ in range(warmups):
        _ = spgemm(A, B)

    var timer = BenchTimer()

    for _ in range(iters):
        var t0 = perf_counter_ns()
        _ = spgemm(A, B)
        var t1 = perf_counter_ns()
        timer.add(t1 - t0)

    var res = timer.compute_stats("Sparse_SpGEMM", "matmul", dim, dim)
    print(res.to_json())


def main() raises:
    var args = argv()
    var gemm_size = 512
    var rows = 50000
    var cols = 50
    var sparse_dim = 10000
    var warmups = 2
    var iters = 5

    if len(args) > 1:
        gemm_size = Int(args[1])
    if len(args) > 2:
        rows = Int(args[2])
    if len(args) > 3:
        warmups = Int(args[3])
    if len(args) > 4:
        iters = Int(args[4])

    run_gemm(gemm_size, warmups, iters)
    run_dense_dot_vec(rows, cols, warmups, iters)
    run_spmv(sparse_dim, 20, warmups, iters)
    run_spmm(min(sparse_dim, 2000), 64, 20, warmups, iters)
    run_spgemm(min(sparse_dim, 2000), 10, warmups, iters)
