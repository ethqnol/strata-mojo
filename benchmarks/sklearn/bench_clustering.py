import sys
import json
import time
import math
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD


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


def make_synthetic_blobs(n_samples, n_features, n_clusters=5, seed=42):
    rng = np.random.RandomState(seed)
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    for k in range(n_clusters):
        centers[k] = (k * 5.0) + (rng.randint(-100, 100, size=n_features)) / 50.0

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples):
        k = i % n_clusters
        noise = (rng.randint(-1000, 1000, size=n_features)) / 1000.0
        X[i] = centers[k] + noise

    return np.ascontiguousarray(X, dtype=np.float64)


def make_synthetic_sparse(rows, cols, nnz_per_row=10, seed=42):
    rng = np.random.RandomState(seed)
    data = []
    indices = []
    indptr = [0]
    for _ in range(rows):
        for _ in range(nnz_per_row):
            c = rng.randint(0, cols)
            val = (rng.randint(1, 1000)) / 100.0
            data.append(val)
            indices.append(c)
        indptr.append(len(data))
    return sp.csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=np.float64)


def run_kmeans(samples, features, warmups, iters):
    X = make_synthetic_blobs(samples, features, n_clusters=8, seed=42)

    for _ in range(warmups):
        km = KMeans(n_clusters=8, init="k-means++", max_iter=100, n_init=1, random_state=42)
        km.fit(X)
        _ = km.predict(X)

    fit_times, pred_times = [], []
    last_inertia = 0.0

    for _ in range(iters):
        km = KMeans(n_clusters=8, init="k-means++", max_iter=100, n_init=1, random_state=42)
        t0 = time.perf_counter_ns()
        km.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = km.predict(X)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_inertia = float(km.inertia_)

    print(json.dumps(compute_stats("KMeans", "fit", samples, features, fit_times, "inertia", last_inertia)))
    print(json.dumps(compute_stats("KMeans", "predict", samples, features, pred_times, "inertia", last_inertia)))


def run_minibatch_kmeans(samples, features, warmups, iters):
    X = make_synthetic_blobs(samples, features, n_clusters=8, seed=42)

    for _ in range(warmups):
        mbk = MiniBatchKMeans(n_clusters=8, batch_size=256, max_iter=100, n_init=1, random_state=42)
        mbk.fit(X)
        _ = mbk.predict(X)

    fit_times, pred_times = [], []
    last_inertia = 0.0

    for _ in range(iters):
        mbk = MiniBatchKMeans(n_clusters=8, batch_size=256, max_iter=100, n_init=1, random_state=42)
        t0 = time.perf_counter_ns()
        mbk.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = mbk.predict(X)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_inertia = float(mbk.inertia_)

    print(json.dumps(compute_stats("MiniBatchKMeans", "fit", samples, features, fit_times, "inertia", last_inertia)))
    print(json.dumps(compute_stats("MiniBatchKMeans", "predict", samples, features, pred_times, "inertia", last_inertia)))


def run_pca(samples, features, warmups, iters):
    X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)
    n_comp = min(5, features)

    for _ in range(warmups):
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        _ = pca.transform(X)

    fit_times, trans_times = [], []
    last_evr = 0.0

    for _ in range(iters):
        pca = PCA(n_components=n_comp)
        t0 = time.perf_counter_ns()
        pca.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = pca.transform(X)
        t3 = time.perf_counter_ns()
        trans_times.append(t3 - t2)

        last_evr = float(pca.explained_variance_ratio_[0])

    print(json.dumps(compute_stats("PCA", "fit", samples, features, fit_times, "explained_variance_ratio_0", last_evr)))
    print(json.dumps(compute_stats("PCA", "transform", samples, features, trans_times, "explained_variance_ratio_0", last_evr)))


def run_truncated_svd(samples, features, warmups, iters):
    X_csr = make_synthetic_sparse(samples, features, nnz_per_row=10, seed=42)
    n_comp = min(5, features)

    for _ in range(warmups):
        svd_model = TruncatedSVD(n_components=n_comp)
        svd_model.fit(X_csr)
        _ = svd_model.transform(X_csr)

    fit_times, trans_times = [], []

    for _ in range(iters):
        svd_model = TruncatedSVD(n_components=n_comp)
        t0 = time.perf_counter_ns()
        svd_model.fit(X_csr)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = svd_model.transform(X_csr)
        t3 = time.perf_counter_ns()
        trans_times.append(t3 - t2)

    print(json.dumps(compute_stats("TruncatedSVD_CSR", "fit", samples, features, fit_times)))
    print(json.dumps(compute_stats("TruncatedSVD_CSR", "transform", samples, features, trans_times)))


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    warmups = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    run_kmeans(samples, features, warmups, iters)
    run_minibatch_kmeans(samples, features, warmups, iters)
    run_dbscan(samples, features, warmups, iters)
    run_pca(samples, features, warmups, iters)
    run_truncated_svd(samples, features, warmups, iters)


def run_dbscan(samples, features, warmups, iters):
    n_dbscan = samples if samples <= 5000 else 5000
    X = make_synthetic_blobs(n_dbscan, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        db = DBSCAN(eps=2.0, min_samples=5)
        db.fit(X)

    fit_times, pred_times = [], []
    last_n_clusters = 0.0

    for _ in range(iters):
        db = DBSCAN(eps=2.0, min_samples=5)
        t0 = time.perf_counter_ns()
        db.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = db.fit_predict(X)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        unique_labels = set(db.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        last_n_clusters = float(n_clusters)

    print(json.dumps(compute_stats("DBSCAN", "fit", n_dbscan, features, fit_times, "n_clusters", last_n_clusters)))
    print(json.dumps(compute_stats("DBSCAN", "predict", n_dbscan, features, pred_times, "n_clusters", last_n_clusters)))


if __name__ == "__main__":
    main()
