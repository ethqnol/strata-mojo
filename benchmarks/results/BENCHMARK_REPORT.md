# Strata vs. Scikit-Learn High-Performance Benchmark Suite

**Date**: 2026-09-09 15:18:49 UTC  
**Workload Scale**: `SMALL`  
**Methodology**: Zero-overhead in-memory allocation, dedicated warmup iterations, nanosecond-precision monotonic timing (`perf_counter_ns`), median execution times across repetitions, strict single-thread environment control.

---

## Executive Summary

- **Total Benchmarks Evaluated**: 69
- **Overall Geometric Mean Speedup**: **0.69x**
- **Strata Faster (≥ 1.05x)**: **23** (33.3%)
- **Equivalent / Parity (0.95x - 1.05x)**: **5**
- **Scikit-Learn Faster (< 0.95x)**: **41**

## 1. Model Training & Fitting Performance (`fit`)

> **Category Summary**: Geometric Mean Speedup = **0.63x** (9/32 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `LinearRegression` | `fit` | 10,000 x 20 | 3.76 ms | 5.02 ms | **1.34x** | 2,659,788 samples/s | r2_score: 1.0000 (Exact) |
| `Ridge` | `fit` | 10,000 x 20 | 3.58 ms | 1.87 ms | 0.52x | 2,796,706 samples/s | r2_score: 1.0000 (Exact) |
| `Lasso` | `fit` | 10,000 x 20 | 3.94 ms | 1.66 ms | 0.42x | 2,538,681 samples/s | r2_score: 1.0000 (Exact) |
| `ElasticNet` | `fit` | 10,000 x 20 | 3.95 ms | 1.72 ms | 0.43x | 2,531,437 samples/s | r2_score: 1.0000 (Exact) |
| `LogisticRegression` | `fit` | 10,000 x 20 | 49.00 ms | 7.65 ms | 0.16x | 204,071 samples/s | accuracy: 1.0000 (Exact) |
| `SGDRegressor` | `fit` | 10,000 x 20 | 17.32 ms | 9.15 ms | 0.53x | 577,250 samples/s | r2_score: 1.0000 (Exact) |
| `SGDClassifier` | `fit` | 10,000 x 20 | 7.17 ms | 6.44 ms | 0.90x | 1,394,358 samples/s | accuracy: 1.0000 (Exact) |
| `LinearSVC` | `fit` | 10,000 x 20 | 76.65 ms | 8.82 ms | 0.12x | 130,457 samples/s | accuracy: 1.0000 (Exact) |
| `LinearSVR` | `fit` | 10,000 x 20 | 2.31 s | 862.09 ms | 0.37x | 4,336 samples/s | r2_score: 1.0000 (Exact) |
| `DecisionTreeClassifier` | `fit` | 10,000 x 15 | 17.98 ms | 21.58 ms | **1.20x** | 556,234 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeRegressor` | `fit` | 10,000 x 15 | 129.34 ms | 146.59 ms | **1.13x** | 77,313 samples/s | r2_score: 0.8938 vs 0.9063 (Δ=1.2e-02) |
| `RandomForestClassifier` | `fit` | 10,000 x 15 | 103.78 ms | 89.54 ms | 0.86x | 96,354 samples/s | accuracy: 1.0000 (Exact) |
| `RandomForestRegressor` | `fit` | 10,000 x 15 | 616.03 ms | 1.83 s | **2.97x** | 16,233 samples/s | r2_score: 0.9084 vs 0.9490 (Δ=4.1e-02) |
| `HistGradientBoostingClassifier` | `fit` | 10,000 x 15 | 204.51 ms | 72.52 ms | 0.35x | 48,896 samples/s | accuracy: 1.0000 (Exact) |
| `HistGradientBoostingRegressor` | `fit` | 10,000 x 15 | 208.51 ms | 140.92 ms | 0.68x | 47,960 samples/s | r2_score: 0.8080 vs 0.9082 (Δ=1.0e-01) |
| `KMeans` | `fit` | 10,000 x 20 | 7.26 ms | 7.14 ms | 0.98x (parity) | 1,378,300 samples/s | inertia: 413246.0183 vs 66500.9246 (Δ=3.5e+05) |
| `MiniBatchKMeans` | `fit` | 10,000 x 20 | 18.21 ms | 15.63 ms | 0.86x | 549,079 samples/s | inertia: 66416.7548 vs 66539.5165 (Δ=1.2e+02) |
| `DBSCAN` | `fit` | 5,000 x 20 | 1.39 s | 83.39 ms | 0.06x | 3,596 samples/s | n_clusters: 46.0000 vs 30.0000 (Δ=1.6e+01) |
| `PCA` | `fit` | 10,000 x 20 | 6.52 ms | 1.27 ms | 0.20x | 1,534,886 samples/s | explained_variance_ratio_0: 0.9788 vs 0.9784 (Δ=4.4e-04) |
| `TruncatedSVD_CSR` | `fit` | 10,000 x 20 | 8.21 ms | 16.22 ms | **1.98x** | 1,217,752 samples/s | Exact Match |
| `NearestNeighbors` | `fit` | 5,000 x 15 | 442.8 µs | 434.8 µs | 0.98x (parity) | 11,291,831 samples/s | Exact Match |
| `KNeighborsClassifier` | `fit` | 5,000 x 15 | 616.6 µs | 850.4 µs | **1.38x** | 8,109,353 samples/s | accuracy: 1.0000 (Exact) |
| `KNeighborsRegressor` | `fit` | 5,000 x 15 | 609.2 µs | 342.2 µs | 0.56x | 8,207,499 samples/s | r2_score: 0.8783 vs 0.8887 (Δ=1.0e-02) |
| `StandardScaler` | `fit` | 50,000 x 30 | 11.51 ms | 11.93 ms | 1.04x (parity) | 4,342,167 samples/s | Exact Match |
| `MinMaxScaler` | `fit` | 50,000 x 30 | 8.68 ms | 7.87 ms | 0.91x | 5,762,267 samples/s | Exact Match |
| `RobustScaler` | `fit` | 50,000 x 30 | 158.71 ms | 65.20 ms | 0.41x | 315,041 samples/s | Exact Match |
| `PolynomialFeatures_d2` | `fit` | 50,000 x 10 | 769.4 µs | 1.00 ms | **1.31x** | 64,989,166 samples/s | Exact Match |
| `GaussianNB` | `fit` | 20,000 x 30 | 12.18 ms | 8.22 ms | 0.67x | 1,641,392 samples/s | accuracy: 1.0000 (Exact) |
| `MultinomialNB` | `fit` | 20,000 x 30 | 15.97 ms | 3.91 ms | 0.25x | 1,252,372 samples/s | accuracy: 1.0000 (Exact) |
| `MultinomialNB_CSR` | `fit` | 20,000 x 30 | 2.30 ms | 3.65 ms | **1.58x** | 8,677,170 samples/s | accuracy: 0.3520 vs 0.3499 (Δ=2.1e-03) |
| `BernoulliNB` | `fit` | 20,000 x 30 | 6.55 ms | 12.35 ms | **1.88x** | 3,053,097 samples/s | accuracy: 0.9944 vs 0.9948 (Δ=3.0e-04) |
| `ComplementNB` | `fit` | 20,000 x 30 | 14.50 ms | 3.92 ms | 0.27x | 1,379,750 samples/s | accuracy: 1.0000 (Exact) |

## 2. Model Inference & Transformation Throughput (`predict`, `transform`, `query`)

> **Category Summary**: Geometric Mean Speedup = **0.79x** (13/32 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `LinearRegression` | `predict` | 10,000 x 20 | 437.6 µs | 292.8 µs | 0.67x | 22,850,980 samples/s | r2_score: 1.0000 (Exact) |
| `Ridge` | `predict` | 10,000 x 20 | 437.8 µs | 241.6 µs | 0.55x | 22,841,011 samples/s | r2_score: 1.0000 (Exact) |
| `Lasso` | `predict` | 10,000 x 20 | 437.3 µs | 239.0 µs | 0.55x | 22,869,270 samples/s | r2_score: 1.0000 (Exact) |
| `ElasticNet` | `predict` | 10,000 x 20 | 436.1 µs | 248.9 µs | 0.57x | 22,928,628 samples/s | r2_score: 1.0000 (Exact) |
| `LogisticRegression` | `predict` | 10,000 x 20 | 935.3 µs | 439.5 µs | 0.47x | 10,691,539 samples/s | accuracy: 1.0000 (Exact) |
| `SGDRegressor` | `predict` | 10,000 x 20 | 449.7 µs | 333.2 µs | 0.74x | 22,236,009 samples/s | r2_score: 1.0000 (Exact) |
| `SGDClassifier` | `predict` | 10,000 x 20 | 989.9 µs | 435.5 µs | 0.44x | 10,101,928 samples/s | accuracy: 1.0000 (Exact) |
| `LinearSVC` | `predict` | 10,000 x 20 | 1.50 ms | 820.3 µs | 0.55x | 6,681,245 samples/s | accuracy: 1.0000 (Exact) |
| `LinearSVR` | `predict` | 10,000 x 20 | 1.02 ms | 734.7 µs | 0.72x | 9,791,068 samples/s | r2_score: 1.0000 (Exact) |
| `DecisionTreeClassifier` | `predict` | 10,000 x 15 | 341.7 µs | 507.9 µs | **1.49x** | 29,262,954 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeRegressor` | `predict` | 10,000 x 15 | 1.18 ms | 1.88 ms | **1.58x** | 8,441,989 samples/s | r2_score: 0.8938 vs 0.9063 (Δ=1.2e-02) |
| `RandomForestClassifier` | `predict` | 10,000 x 15 | 8.17 ms | 3.33 ms | 0.41x | 1,223,301 samples/s | accuracy: 1.0000 (Exact) |
| `RandomForestRegressor` | `predict` | 10,000 x 15 | 26.17 ms | 32.45 ms | **1.24x** | 382,165 samples/s | r2_score: 0.9084 vs 0.9490 (Δ=4.1e-02) |
| `HistGradientBoostingClassifier` | `predict` | 10,000 x 15 | 10.97 ms | 13.29 ms | **1.21x** | 911,764 samples/s | accuracy: 1.0000 (Exact) |
| `HistGradientBoostingRegressor` | `predict` | 10,000 x 15 | 10.05 ms | 23.92 ms | **2.38x** | 994,910 samples/s | r2_score: 0.8080 vs 0.9082 (Δ=1.0e-01) |
| `KMeans` | `predict` | 10,000 x 20 | 1.01 ms | 684.1 µs | 0.68x | 9,873,481 samples/s | inertia: 413246.0183 vs 66500.9246 (Δ=3.5e+05) |
| `MiniBatchKMeans` | `predict` | 10,000 x 20 | 1.01 ms | 647.4 µs | 0.64x | 9,936,891 samples/s | inertia: 66416.7548 vs 66539.5165 (Δ=1.2e+02) |
| `DBSCAN` | `predict` | 5,000 x 20 | 30.04 ms | 83.34 ms | **2.77x** | 166,458 samples/s | n_clusters: 46.0000 vs 30.0000 (Δ=1.6e+01) |
| `PCA` | `transform` | 10,000 x 20 | 1.98 ms | 400.6 µs | 0.20x | 5,062,514 samples/s | explained_variance_ratio_0: 0.9788 vs 0.9784 (Δ=4.4e-04) |
| `TruncatedSVD_CSR` | `transform` | 10,000 x 20 | 1.78 ms | 625.8 µs | 0.35x | 5,619,218 samples/s | Exact Match |
| `NearestNeighbors` | `kneighbors` | 500 x 15 | 11.36 ms | 13.67 ms | **1.20x** | 44,021 samples/s | Exact Match |
| `KNeighborsClassifier` | `predict` | 500 x 15 | 11.53 ms | 14.33 ms | **1.24x** | 43,371 samples/s | accuracy: 1.0000 (Exact) |
| `KNeighborsRegressor` | `predict` | 500 x 15 | 11.34 ms | 13.59 ms | **1.20x** | 44,107 samples/s | r2_score: 0.8783 vs 0.8887 (Δ=1.0e-02) |
| `StandardScaler` | `transform` | 50,000 x 30 | 4.14 ms | 4.98 ms | **1.20x** | 12,069,674 samples/s | Exact Match |
| `MinMaxScaler` | `transform` | 50,000 x 30 | 4.16 ms | 4.57 ms | **1.10x** | 12,014,812 samples/s | Exact Match |
| `RobustScaler` | `transform` | 50,000 x 30 | 4.46 ms | 5.00 ms | **1.12x** | 11,221,330 samples/s | Exact Match |
| `PolynomialFeatures_d2` | `transform` | 50,000 x 10 | 22.55 ms | 21.95 ms | 0.97x (parity) | 2,217,353 samples/s | Exact Match |
| `GaussianNB` | `predict` | 20,000 x 30 | 10.77 ms | 8.53 ms | 0.79x | 1,856,484 samples/s | accuracy: 1.0000 (Exact) |
| `MultinomialNB` | `predict` | 20,000 x 30 | 3.16 ms | 1.35 ms | 0.43x | 6,330,805 samples/s | accuracy: 1.0000 (Exact) |
| `MultinomialNB_CSR` | `predict` | 20,000 x 30 | 2.31 ms | 1.36 ms | 0.59x | 8,660,457 samples/s | accuracy: 0.3520 vs 0.3499 (Δ=2.1e-03) |
| `BernoulliNB` | `predict` | 20,000 x 30 | 6.85 ms | 9.77 ms | **1.42x** | 2,917,674 samples/s | accuracy: 0.9944 vs 0.9948 (Δ=3.0e-04) |
| `ComplementNB` | `predict` | 20,000 x 30 | 2.96 ms | 1.20 ms | 0.40x | 6,746,200 samples/s | accuracy: 1.0000 (Exact) |

## 3. Low-Level Linear Algebra & Sparse Kernels (`gemm`, `spmv`, `spmm`)

> **Category Summary**: Geometric Mean Speedup = **0.55x** (1/5 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Dense_GEMM` | `matmul` | 512 x 512 | 10.30 ms | 10.36 ms | 1.01x (parity) | 49,695 samples/s | Exact Match |
| `Dense_Dot_Vec` | `matvec` | 50,000 x 50 | 723.0 µs | 1.02 ms | **1.41x** | 69,153,232 samples/s | Exact Match |
| `Sparse_SpMV` | `matvec` | 10,000 x 10000 | 948.6 µs | 247.7 µs | 0.26x | 10,541,918 samples/s | Exact Match |
| `Sparse_SpMM` | `matmul` | 2,000 x 64 | 3.86 ms | 1.02 ms | 0.26x | 518,270 samples/s | Exact Match |
| `Sparse_SpGEMM` | `matmul` | 2,000 x 2000 | 3.05 ms | 1.60 ms | 0.52x | 656,384 samples/s | Exact Match |

