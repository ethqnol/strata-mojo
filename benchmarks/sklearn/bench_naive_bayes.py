import sys
import json
import time
import math
import numpy as np
import scipy.sparse as sp
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
)
from sklearn.metrics import accuracy_score


def compute_stats(benchmark, phase, samples, features, times_ns, metric_name="none", metric_val=0.0):
    n = len(times_ns)
    ms_list = [t / 1_000_000.0 for t in times_ns]
    mean_ms = sum(ms_list) / n
    min_ms = min(ms_list)
    max_ms = max(ms_list)
    sorted_ms = sorted(ms_list)
    if n % 2 == 1:
        median_ms = sorted_ms[n // 2]
    else:
        median_ms = (sorted_ms[n // 2 - 1] + sorted_ms[n // 2]) / 2.0

    sum_sq_diff = sum((x - mean_ms) ** 2 for x in ms_list)
    std_ms = math.sqrt(sum_sq_diff / n)
    throughput = (samples / median_ms) * 1000.0 if median_ms > 0 else 0.0

    return {
        "benchmark": benchmark,
        "phase": phase,
        "samples": samples,
        "features": features,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "throughput_samples_per_sec": throughput,
        "metric_name": metric_name,
        "metric_val": metric_val,
        "iterations": n,
    }


def make_synthetic_classification(n_samples, n_features, n_classes=2, seed=42):
    rng = np.random.RandomState(seed)
    centroids = np.zeros((n_classes, n_features), dtype=np.float64)
    for c in range(n_classes):
        centroids[c] = (c * 3.0) + (rng.randint(-100, 100, size=n_features)) / 100.0

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        c = i % n_classes
        y[i] = c
        offset = (rng.randint(-1000, 1000, size=n_features)) / 1000.0
        X[i] = centroids[c] + offset

    return np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.int32)


def make_synthetic_counts(n_samples, n_features, n_classes=2, seed=42):
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        c = i % n_classes
        y[i] = c
        for j in range(n_features):
            base_rate = 5 if (j % n_classes) == c else 1
            X[i, j] = float(rng.randint(0, base_rate * 3 + 1))
    return np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.int32)


def make_synthetic_sparse(rows, cols, nnz_per_row=10, seed=42):
    rng = np.random.RandomState(seed)
    data = []
    indices = []
    indptr = [0]
    for _ in range(rows):
        for _ in range(nnz_per_row):
            c = rng.randint(0, cols)
            val = float(rng.randint(1, 10))
            data.append(val)
            indices.append(c)
        indptr.append(len(data))
    return sp.csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=np.float64)


def run_gaussian_nb(samples, features, warmups, iters):
    X, y = make_synthetic_classification(samples, features, n_classes=3, seed=42)

    for _ in range(warmups):
        gnb = GaussianNB()
        gnb.fit(X, y)
        _ = gnb.predict(X)

    fit_times, pred_times = [], []
    last_acc = 0.0

    for _ in range(iters):
        gnb = GaussianNB()
        t0 = time.perf_counter_ns()
        gnb.fit(X, y)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        preds = gnb.predict(X)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_acc = float(accuracy_score(y, preds))

    print(json.dumps(compute_stats("GaussianNB", "fit", samples, features, fit_times, "accuracy", last_acc)))
    print(json.dumps(compute_stats("GaussianNB", "predict", samples, features, pred_times, "accuracy", last_acc)))


def run_multinomial_nb(samples, features, warmups, iters):
    X, y = make_synthetic_counts(samples, features, n_classes=3, seed=42)

    for _ in range(warmups):
        mnb = MultinomialNB()
        mnb.fit(X, y)
        _ = mnb.predict(X)

    fit_times, pred_times = [], []
    last_acc = 0.0

    for _ in range(iters):
        mnb = MultinomialNB()
        t0 = time.perf_counter_ns()
        mnb.fit(X, y)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        preds = mnb.predict(X)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_acc = float(accuracy_score(y, preds))

    print(json.dumps(compute_stats("MultinomialNB", "fit", samples, features, fit_times, "accuracy", last_acc)))
    print(json.dumps(compute_stats("MultinomialNB", "predict", samples, features, pred_times, "accuracy", last_acc)))


def run_multinomial_nb_csr(samples, features, warmups, iters):
    nnz = features // 4 if features >= 4 else 2
    X_csr = make_synthetic_sparse(samples, features, nnz_per_row=nnz, seed=42)
    _, y = make_synthetic_counts(samples, features, n_classes=3, seed=42)

    for _ in range(warmups):
        mnb = MultinomialNB()
        mnb.fit(X_csr, y)
        _ = mnb.predict(X_csr)

    fit_times, pred_times = [], []
    last_acc = 0.0

    for _ in range(iters):
        mnb = MultinomialNB()
        t0 = time.perf_counter_ns()
        mnb.fit(X_csr, y)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        preds = mnb.predict(X_csr)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_acc = float(accuracy_score(y, preds))

    print(json.dumps(compute_stats("MultinomialNB_CSR", "fit", samples, features, fit_times, "accuracy", last_acc)))
    print(json.dumps(compute_stats("MultinomialNB_CSR", "predict", samples, features, pred_times, "accuracy", last_acc)))


def run_bernoulli_nb(samples, features, warmups, iters):
    X, y = make_synthetic_counts(samples, features, n_classes=3, seed=42)

    for _ in range(warmups):
        bnb = BernoulliNB(binarize=2.0)
        bnb.fit(X, y)
        _ = bnb.predict(X)

    fit_times, pred_times = [], []
    last_acc = 0.0

    for _ in range(iters):
        bnb = BernoulliNB(binarize=2.0)
        t0 = time.perf_counter_ns()
        bnb.fit(X, y)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        preds = bnb.predict(X)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_acc = float(accuracy_score(y, preds))

    print(json.dumps(compute_stats("BernoulliNB", "fit", samples, features, fit_times, "accuracy", last_acc)))
    print(json.dumps(compute_stats("BernoulliNB", "predict", samples, features, pred_times, "accuracy", last_acc)))


def run_complement_nb(samples, features, warmups, iters):
    X, y = make_synthetic_counts(samples, features, n_classes=3, seed=42)

    for _ in range(warmups):
        cnb = ComplementNB(norm=True)
        cnb.fit(X, y)
        _ = cnb.predict(X)

    fit_times, pred_times = [], []
    last_acc = 0.0

    for _ in range(iters):
        cnb = ComplementNB(norm=True)
        t0 = time.perf_counter_ns()
        cnb.fit(X, y)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        preds = cnb.predict(X)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_acc = float(accuracy_score(y, preds))

    print(json.dumps(compute_stats("ComplementNB", "fit", samples, features, fit_times, "accuracy", last_acc)))
    print(json.dumps(compute_stats("ComplementNB", "predict", samples, features, pred_times, "accuracy", last_acc)))


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmups = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    run_gaussian_nb(samples, features, warmups, iters)
    run_multinomial_nb(samples, features, warmups, iters)
    run_multinomial_nb_csr(samples, features, warmups, iters)
    run_bernoulli_nb(samples, features, warmups, iters)
    run_complement_nb(samples, features, warmups, iters)


if __name__ == "__main__":
    main()
