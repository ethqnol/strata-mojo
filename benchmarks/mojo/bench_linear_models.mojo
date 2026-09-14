from std.time import perf_counter_ns
from std.sys import argv
from strata import (
    Matrix,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    SGDRegressor,
    SGDClassifier,
    LinearSVC,
    LinearSVR,
    r2_score,
    accuracy_score,
)
from benchmarks.mojo.bench_utils import (
    BenchTimer,
    make_synthetic_regression,
    make_synthetic_classification,
)


def run_linear_regression(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    # Warmup
    for _ in range(warmups):
        var reg = LinearRegression(solver="cholesky")
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = LinearRegression(solver="cholesky")
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "LinearRegression",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "LinearRegression",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_ridge(samples: Int, features: Int, warmups: Int, iters: Int) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = Ridge(alpha=1.0, solver="cholesky")
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = Ridge(alpha=1.0, solver="cholesky")
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "Ridge",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "Ridge",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_lasso(samples: Int, features: Int, warmups: Int, iters: Int) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = Lasso(alpha=0.1, max_iter=200, tol=1e-4)
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = Lasso(alpha=0.1, max_iter=200, tol=1e-4)
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "Lasso",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "Lasso",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_elastic_net(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=200, tol=1e-4)
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=200, tol=1e-4)
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "ElasticNet",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "ElasticNet",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_logistic_regression(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=2, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var clf = LogisticRegression(max_iter=100, tol=1e-4)
        clf.fit(X, y)
        _ = clf.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var clf = LogisticRegression(max_iter=100, tol=1e-4)
        var t0 = perf_counter_ns()
        clf.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = clf.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "LogisticRegression",
        "fit",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    var pred_res = pred_timer.compute_stats(
        "LogisticRegression",
        "predict",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_sgd_regressor(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = SGDRegressor(max_iter=100, random_state=42)
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = SGDRegressor(max_iter=100, random_state=42)
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "SGDRegressor",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "SGDRegressor",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_sgd_classifier(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=2, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var clf = SGDClassifier(max_iter=100, random_state=42)
        clf.fit(X, y)
        _ = clf.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var clf = SGDClassifier(max_iter=100, random_state=42)
        var t0 = perf_counter_ns()
        clf.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = clf.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "SGDClassifier",
        "fit",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    var pred_res = pred_timer.compute_stats(
        "SGDClassifier",
        "predict",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def main() raises:
    var args = argv()
    var samples = 20000
    var features = 20
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

    run_linear_regression(samples, features, warmups, iters)
    run_ridge(samples, features, warmups, iters)
    run_lasso(samples, features, warmups, iters)
    run_elastic_net(samples, features, warmups, iters)
    run_logistic_regression(samples, features, warmups, iters)
    run_sgd_regressor(samples, features, warmups, iters)
    run_sgd_classifier(samples, features, warmups, iters)
    run_linear_svc(samples, features, warmups, iters)
    run_linear_svr(samples, features, warmups, iters)


def run_linear_svc(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=2, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var clf = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X, y)
        _ = clf.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var clf = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        var t0 = perf_counter_ns()
        clf.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = clf.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "LinearSVC",
        "fit",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    var pred_res = pred_timer.compute_stats(
        "LinearSVC",
        "predict",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_linear_svr(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = LinearSVR(C=1.0, epsilon=0.1, max_iter=1000, random_state=42)
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = LinearSVR(C=1.0, epsilon=0.1, max_iter=1000, random_state=42)
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "LinearSVR",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "LinearSVR",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())
