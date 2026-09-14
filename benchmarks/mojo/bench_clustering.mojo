from std.time import perf_counter_ns
from std.sys import argv
from strata import (
    Matrix,
    KMeans,
    MiniBatchKMeans,
    DBSCAN,
    PCA,
    TruncatedSVD,
)
from benchmarks.mojo.bench_utils import (
    BenchTimer,
    make_synthetic_blobs,
    make_synthetic_sparse,
)


def run_kmeans(samples: Int, features: Int, warmups: Int, iters: Int) raises:
    var X = make_synthetic_blobs(samples, features, n_clusters=8, seed=42)

    for _ in range(warmups):
        var km = KMeans(
            n_clusters=8,
            init="k-means++",
            max_iter=100,
            n_init=1,
            random_state=42,
        )
        km.fit(X)
        _ = km.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_inertia: Float64 = 0.0

    for _ in range(iters):
        var km = KMeans(
            n_clusters=8,
            init="k-means++",
            max_iter=100,
            n_init=1,
            random_state=42,
        )
        var t0 = perf_counter_ns()
        km.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = km.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_inertia = km.inertia_

    var fit_res = fit_timer.compute_stats(
        "KMeans",
        "fit",
        samples,
        features,
        "inertia",
        last_inertia,
    )
    var pred_res = pred_timer.compute_stats(
        "KMeans",
        "predict",
        samples,
        features,
        "inertia",
        last_inertia,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_minibatch_kmeans(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var X = make_synthetic_blobs(samples, features, n_clusters=8, seed=42)

    for _ in range(warmups):
        var mbk = MiniBatchKMeans(
            n_clusters=8, batch_size=256, max_iter=100, random_state=42
        )
        mbk.fit(X)
        _ = mbk.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_inertia: Float64 = 0.0

    for _ in range(iters):
        var mbk = MiniBatchKMeans(
            n_clusters=8, batch_size=256, max_iter=100, random_state=42
        )
        var t0 = perf_counter_ns()
        mbk.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = mbk.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_inertia = mbk.inertia_

    var fit_res = fit_timer.compute_stats(
        "MiniBatchKMeans",
        "fit",
        samples,
        features,
        "inertia",
        last_inertia,
    )
    var pred_res = pred_timer.compute_stats(
        "MiniBatchKMeans",
        "predict",
        samples,
        features,
        "inertia",
        last_inertia,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_pca(samples: Int, features: Int, warmups: Int, iters: Int) raises:
    var X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)
    var n_comp = 5
    if n_comp > features:
        n_comp = features

    for _ in range(warmups):
        var pca = PCA(n_components=n_comp)
        pca.fit(X)
        _ = pca.transform(X)

    var fit_timer = BenchTimer()
    var trans_timer = BenchTimer()
    var last_evr: Float64 = 0.0

    for _ in range(iters):
        var pca = PCA(n_components=n_comp)
        var t0 = perf_counter_ns()
        pca.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = pca.transform(X)
        var t3 = perf_counter_ns()
        trans_timer.add(t3 - t2)

        last_evr = pca.explained_variance_ratio_[0]

    var fit_res = fit_timer.compute_stats(
        "PCA",
        "fit",
        samples,
        features,
        "explained_variance_ratio_0",
        last_evr,
    )
    var trans_res = trans_timer.compute_stats(
        "PCA",
        "transform",
        samples,
        features,
        "explained_variance_ratio_0",
        last_evr,
    )
    print(fit_res.to_json())
    print(trans_res.to_json())


def run_truncated_svd(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var X_csr = make_synthetic_sparse(
        samples, features, nnz_per_row=10, seed=42
    )
    var n_comp = 5
    if n_comp > features:
        n_comp = features

    for _ in range(warmups):
        var svd_model = TruncatedSVD(n_components=n_comp)
        svd_model.fit(X_csr)
        _ = svd_model.transform(X_csr)

    var fit_timer = BenchTimer()
    var trans_timer = BenchTimer()

    for _ in range(iters):
        var svd_model = TruncatedSVD(n_components=n_comp)
        var t0 = perf_counter_ns()
        svd_model.fit(X_csr)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = svd_model.transform(X_csr)
        var t3 = perf_counter_ns()
        trans_timer.add(t3 - t2)

    var fit_res = fit_timer.compute_stats(
        "TruncatedSVD_CSR",
        "fit",
        samples,
        features,
    )
    var trans_res = trans_timer.compute_stats(
        "TruncatedSVD_CSR",
        "transform",
        samples,
        features,
    )
    print(fit_res.to_json())
    print(trans_res.to_json())


def main() raises:
    var args = argv()
    var samples = 10000
    var features = 20
    var warmups = 1
    var iters = 3

    if len(args) > 1:
        samples = Int(args[1])
    if len(args) > 2:
        features = Int(args[2])
    if len(args) > 3:
        warmups = Int(args[3])
    if len(args) > 4:
        iters = Int(args[4])

    run_kmeans(samples, features, warmups, iters)
    run_minibatch_kmeans(samples, features, warmups, iters)
    run_dbscan(samples, features, warmups, iters)
    run_pca(samples, features, warmups, iters)
    run_truncated_svd(samples, features, warmups, iters)


def run_dbscan(samples: Int, features: Int, warmups: Int, iters: Int) raises:
    var n_dbscan = samples if samples <= 5000 else 5000
    var X = make_synthetic_blobs(n_dbscan, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        var db = DBSCAN(eps=2.0, min_samples=5)
        db.fit(X)
        _ = db.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_n_clusters: Float64 = 0.0

    for _ in range(iters):
        var db = DBSCAN(eps=2.0, min_samples=5)
        var t0 = perf_counter_ns()
        db.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = db.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_n_clusters = Float64(db.n_clusters_)

    var fit_res = fit_timer.compute_stats(
        "DBSCAN",
        "fit",
        n_dbscan,
        features,
        "n_clusters",
        last_n_clusters,
    )
    var pred_res = pred_timer.compute_stats(
        "DBSCAN",
        "predict",
        n_dbscan,
        features,
        "n_clusters",
        last_n_clusters,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())
