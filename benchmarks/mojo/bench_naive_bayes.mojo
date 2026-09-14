from std.time import perf_counter_ns
from std.sys import argv
from strata import (
    Matrix,
    CSRMatrix,
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
    accuracy_score,
)
from benchmarks.mojo.bench_utils import (
    BenchTimer,
    make_synthetic_classification,
    make_synthetic_counts,
    make_synthetic_sparse,
)


def run_gaussian_nb(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=3, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var gnb = GaussianNB()
        gnb.fit(X, y)
        _ = gnb.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var gnb = GaussianNB()
        var t0 = perf_counter_ns()
        gnb.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = gnb.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "GaussianNB", "fit", samples, features, "accuracy", last_acc
    )
    var pred_res = pred_timer.compute_stats(
        "GaussianNB", "predict", samples, features, "accuracy", last_acc
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_multinomial_nb(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_counts(samples, features, n_classes=3, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var mnb = MultinomialNB()
        mnb.fit(X, y)
        _ = mnb.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var mnb = MultinomialNB()
        var t0 = perf_counter_ns()
        mnb.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = mnb.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "MultinomialNB", "fit", samples, features, "accuracy", last_acc
    )
    var pred_res = pred_timer.compute_stats(
        "MultinomialNB", "predict", samples, features, "accuracy", last_acc
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_multinomial_nb_csr(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var X_csr = make_synthetic_sparse(
        samples,
        features,
        nnz_per_row=features // 4 if features >= 4 else 2,
        seed=42,
    )
    var data = make_synthetic_counts(samples, features, n_classes=3, seed=42)
    var y = data[1].copy()

    for _ in range(warmups):
        var mnb = MultinomialNB()
        mnb.fit(X_csr, y)
        _ = mnb.predict(X_csr)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var mnb = MultinomialNB()
        var t0 = perf_counter_ns()
        mnb.fit(X_csr, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = mnb.predict(X_csr)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "MultinomialNB_CSR", "fit", samples, features, "accuracy", last_acc
    )
    var pred_res = pred_timer.compute_stats(
        "MultinomialNB_CSR", "predict", samples, features, "accuracy", last_acc
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_bernoulli_nb(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_counts(samples, features, n_classes=3, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var bnb = BernoulliNB(binarize=2.0)
        bnb.fit(X, y)
        _ = bnb.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var bnb = BernoulliNB(binarize=2.0)
        var t0 = perf_counter_ns()
        bnb.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = bnb.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "BernoulliNB", "fit", samples, features, "accuracy", last_acc
    )
    var pred_res = pred_timer.compute_stats(
        "BernoulliNB", "predict", samples, features, "accuracy", last_acc
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_complement_nb(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_counts(samples, features, n_classes=3, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var cnb = ComplementNB(norm=True)
        cnb.fit(X, y)
        _ = cnb.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var cnb = ComplementNB(norm=True)
        var t0 = perf_counter_ns()
        cnb.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = cnb.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "ComplementNB", "fit", samples, features, "accuracy", last_acc
    )
    var pred_res = pred_timer.compute_stats(
        "ComplementNB", "predict", samples, features, "accuracy", last_acc
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def main() raises:
    var args = argv()
    var samples = 20000
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

    run_gaussian_nb(samples, features, warmups, iters)
    run_multinomial_nb(samples, features, warmups, iters)
    run_multinomial_nb_csr(samples, features, warmups, iters)
    run_bernoulli_nb(samples, features, warmups, iters)
    run_complement_nb(samples, features, warmups, iters)
