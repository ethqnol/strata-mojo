from std.time import perf_counter_ns
from std.sys import argv
from strata import (
    Matrix,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
)
from benchmarks.mojo.bench_utils import (
    BenchTimer,
    make_synthetic_blobs,
)


def run_standard_scaler(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        var scaler = StandardScaler()
        scaler.fit(X)
        _ = scaler.transform(X)

    var fit_timer = BenchTimer()
    var trans_timer = BenchTimer()

    for _ in range(iters):
        var scaler = StandardScaler()
        var t0 = perf_counter_ns()
        scaler.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = scaler.transform(X)
        var t3 = perf_counter_ns()
        trans_timer.add(t3 - t2)

    var fit_res = fit_timer.compute_stats(
        "StandardScaler", "fit", samples, features
    )
    var trans_res = trans_timer.compute_stats(
        "StandardScaler", "transform", samples, features
    )
    print(fit_res.to_json())
    print(trans_res.to_json())


def run_minmax_scaler(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        var scaler = MinMaxScaler()
        scaler.fit(X)
        _ = scaler.transform(X)

    var fit_timer = BenchTimer()
    var trans_timer = BenchTimer()

    for _ in range(iters):
        var scaler = MinMaxScaler()
        var t0 = perf_counter_ns()
        scaler.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = scaler.transform(X)
        var t3 = perf_counter_ns()
        trans_timer.add(t3 - t2)

    var fit_res = fit_timer.compute_stats(
        "MinMaxScaler", "fit", samples, features
    )
    var trans_res = trans_timer.compute_stats(
        "MinMaxScaler", "transform", samples, features
    )
    print(fit_res.to_json())
    print(trans_res.to_json())


def run_robust_scaler(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        var scaler = RobustScaler()
        scaler.fit(X)
        _ = scaler.transform(X)

    var fit_timer = BenchTimer()
    var trans_timer = BenchTimer()

    for _ in range(iters):
        var scaler = RobustScaler()
        var t0 = perf_counter_ns()
        scaler.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = scaler.transform(X)
        var t3 = perf_counter_ns()
        trans_timer.add(t3 - t2)

    var fit_res = fit_timer.compute_stats(
        "RobustScaler", "fit", samples, features
    )
    var trans_res = trans_timer.compute_stats(
        "RobustScaler", "transform", samples, features
    )
    print(fit_res.to_json())
    print(trans_res.to_json())


def run_polynomial_features(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var poly_features = min(features, 10)
    var X = make_synthetic_blobs(samples, poly_features, n_clusters=3, seed=42)

    for _ in range(warmups):
        var poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(X)
        _ = poly.transform(X)

    var fit_timer = BenchTimer()
    var trans_timer = BenchTimer()

    for _ in range(iters):
        var poly = PolynomialFeatures(degree=2, include_bias=True)
        var t0 = perf_counter_ns()
        poly.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = poly.transform(X)
        var t3 = perf_counter_ns()
        trans_timer.add(t3 - t2)

    var fit_res = fit_timer.compute_stats(
        "PolynomialFeatures_d2", "fit", samples, poly_features
    )
    var trans_res = trans_timer.compute_stats(
        "PolynomialFeatures_d2", "transform", samples, poly_features
    )
    print(fit_res.to_json())
    print(trans_res.to_json())


def main() raises:
    var args = argv()
    var samples = 50000
    var features = 30
    var warmups = 2
    var iters = 5

    if len(args) > 1:
        samples = Int(args[1])
    if len(args) > 2:
        features = Int(args[2])
    if len(args) > 3:
        warmups = Int(args[3])
    if len(args) > 4:
        iters = Int(args[4])

    run_standard_scaler(samples, features, warmups, iters)
    run_minmax_scaler(samples, features, warmups, iters)
    run_robust_scaler(samples, features, warmups, iters)
    run_polynomial_features(samples, features, warmups, iters)
