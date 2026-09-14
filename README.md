[![CodeQL](https://github.com/ethqnol/strata-mojo/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ethqnol/strata-mojo/actions/workflows/github-code-scanning/codeql) [![Prefix.dev](https://img.shields.io/badge/prefix.dev-strata-purple)](https://prefix.dev/channels/modular-community/packages/strata) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Documentation](https://img.shields.io/badge/docs-passing-brightgreen?logo=github)](https://ethqnol.github.io/strata-mojo)

# Strata

<p align="center">
  <img src="assets/strata.jpg" alt="Strata — Machine Learning in Pure Mojo" width="70%">
</p>

Strata is a native machine learning and linear algebra library written in [Mojo](https://docs.modular.com/mojo/). It provides scikit-learn-compatible estimators, dense and sparse matrix containers, zero-copy binary serialization, and composable pipelines with native SIMD and LAPACK acceleration.

---

## Installation

### In your Pixi Project

Add Strata directly from the official Modular Community channel on [Prefix.dev](https://prefix.dev/channels/modular-community/packages/strata):

```bash
pixi add strata --channel https://repo.prefix.dev/modular-community
```

Or configure your `pixi.toml`:

```toml
[workspace]
channels = [
    "https://repo.prefix.dev/modular-community",
    "https://conda.modular.com/max",
    "conda-forge"
]

[dependencies]
strata = ">=0.2.0"
mojo = ">=1.0.0"

# Task shortcut to automatically link LAPACK & BLAS shared libraries
[tasks]
start = "mojo run -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack -Xlinker -lblas main.mojo"
```

> [!IMPORTANT]
> **LAPACK & BLAS Linker Flags**: `pixi add strata` automatically installs `liblapack` and `libblas`. When calling LAPACK/BLAS-accelerated routines (`svd`, `qr`, `inv`, `eigh`, `cholesky`, `solve`, `lstsq`, `PCA`, `TruncatedSVD`, `LinearRegression`, or `Ridge`), pass `-Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack -Xlinker -lblas` to `mojo run`, or define a `start` task in your `pixi.toml`.

### For Local Development

```bash
git clone https://github.com/ethqnol/strata-mojo.git
cd Strata
pixi install
```

---

## Performance Benchmarks vs. Scikit-Learn

Strata combines Mojo's compile-time metaprogramming and hardware SIMD vectorization with zero-copy memory layouts to outperform Scikit-Learn across standard workloads while guaranteeing **100% bit-for-bit mathematical parity**:

| Domain | Estimator / Routine | Phase | Strata Median | Scikit-Learn Median | Speedup | Quality / Parity |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Trees & Ensembles**| `RandomForestRegressor` | `fit` | **616.03 ms** | 1.83 s | **2.97x faster** | Exact Parity ($R^2 = 0.91$) |
| | `HistGradientBoostingRegressor` | `predict` | **10.05 ms** | 23.92 ms | **2.38x faster** | Exact Parity ($R^2 = 0.81$) |
| | `DecisionTreeRegressor` | `predict` | **1.18 ms** | 1.88 ms | **1.58x faster** | Exact Parity ($R^2 = 0.89$) |
| | `DecisionTreeClassifier` | `predict` | **341.7 µs** | 507.9 µs | **1.49x faster** | Exact Parity (`1.0000`) |
| | `DecisionTreeClassifier` | `fit` | **17.98 ms** | 21.58 ms | **1.20x faster** | Exact Parity (`1.0000`) |
| **Clustering** | `DBSCAN` | `predict` | **30.04 ms** | 83.34 ms | **2.77x faster** | Exact Match |
| | `TruncatedSVD_CSR` | `fit` | **8.21 ms** | 16.22 ms | **1.98x faster** | Exact Match |
| | `KMeans` | `fit` | **7.26 ms** | 7.14 ms | **≈ parity (1.02x)**| Exact Match |
| **Naive Bayes** | `BernoulliNB` | `fit` | **6.55 ms** | 12.35 ms | **1.88x faster** | Exact Parity (`0.9944`) |
| | `MultinomialNB_CSR` | `fit` | **2.30 ms** | 3.65 ms | **1.58x faster** | Exact Parity (`0.3520`) |
| | `BernoulliNB` | `predict` | **6.85 ms** | 9.77 ms | **1.42x faster** | Exact Parity (`0.9944`) |
| **Linear Models** | `LinearRegression` | `fit` | **3.76 ms** | 5.02 ms | **1.34x faster** | Exact Parity ($R^2 = 1.0000$) |
| **Nearest Neighbors**| `KNeighborsClassifier` | `fit` | **616.6 µs** | 850.4 µs | **1.38x faster** | Exact Parity (`1.0000`) |
| | `KNeighborsClassifier` | `predict` | **11.53 ms** | 14.33 ms | **1.24x faster** | Exact Parity (`1.0000`) |
| | `NearestNeighbors` | `kneighbors` | **11.36 ms** | 13.67 ms | **1.20x faster** | Exact Match |
| **Preprocessing** | `PolynomialFeatures` | `fit` | **769.4 µs** | 1.00 ms | **1.31x faster** | Exact Match |
| | `StandardScaler` | `transform` | **4.14 ms** | 4.98 ms | **1.20x faster** | Exact Match |
| | `RobustScaler` | `transform` | **4.46 ms** | 5.00 ms | **1.12x faster** | Exact Match |
| | `MinMaxScaler` | `transform` | **4.16 ms** | 4.57 ms | **1.10x faster** | Exact Match |
| **Linear Algebra** | `Dense_Dot_Vec` | `matvec` | **723.0 µs** | 1.02 ms | **1.41x faster** | Exact Match |
| | `Dense_GEMM` | `matmul` | **10.30 ms** | 10.36 ms | **≈ parity (1.01x)**| Exact Match |

*For complete reproducible benchmark methodology, charts, and large-scale runs, see [BENCHMARK_REPORT.md](./benchmarks/results/BENCHMARK_REPORT.md).*

---

## Quick Start

> [!NOTE]
> **Execution Modes**:
> - **Pure Mojo Routines (Zero External Dependencies)**: Decision Trees, Random Forests, Gradient Boosting, SGD, Coordinate Descent (Lasso/ElasticNet), KMeans, Scalers, Encoders, ColumnTransformer, KDTree, Metrics, Model Selection, and Serialization. Run directly with `mojo run main.mojo`.
> - **LAPACK & BLAS Accelerated Solvers**: `LinearRegression` (analytical solvers), `Ridge`, `LogisticRegression` (GEMM-accelerated), `NearestNeighbors` / `KNeighbors` (GEMM distance matrix), `PCA`, `TruncatedSVD`, and core matrix factorizations (`svd`, `qr`, `inv`, `eigh`, `cholesky`, `solve`, `lstsq`, `gemm`). Run with `pixi run start` or pass `-Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack -Xlinker -lblas`.

### Linear Algebra & Matrix Operations

```mojo
from strata import Matrix, gemm, svd, eigh, solve

def main() raises:
    # Dense matrices
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 4.0; A[0, 1] = 2.0
    A[1, 0] = 2.0; A[1, 1] = 3.0

    var B = Matrix[DType.float64].eye(2)

    # Matrix multiplication
    var C = gemm(A, B)

    # Factorizations (LAPACK)
    var svd_res = svd(A)
    var eig_res = eigh(A)

    # Linear system solve: A * x = b
    var b: List[Scalar[DType.float64]] = [8.0, 7.0]
    var x = solve(A, b)
```

### Pipelines & Estimators

```mojo
from strata import (
    Matrix,
    StandardScaler,
    PCA,
    LinearRegression,
    PipelineTransformer,
    PipelineRegressor,
)

def main() raises:
    var X = Matrix[DType.float64](5, 3, 0)
    for r in range(5):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = Float64((r + 1) * 2)
        X[r, 2] = Float64((r + 1) * 3)

    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0]

    # Build a StandardScaler -> PCA -> LinearRegression pipeline
    var scaler = StandardScaler()
    var pca = PCA(n_components=2)
    var prep = PipelineTransformer((scaler^, pca^))
    var reg = LinearRegression(solver="cholesky")
    var pipe = PipelineRegressor(prep^, reg^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)
```

### Clustering (KMeans & DBSCAN)

```mojo
from strata import Matrix, KMeans, MiniBatchKMeans, DBSCAN

def main() raises:
    var X = Matrix[DType.float64](100, 2, 0)

    # 1. Geometric K-Means with k-means++ initialization
    var kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
    kmeans.fit(X)
    var km_labels = kmeans.predict(X)

    # 2. Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    var dbscan = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
    var db_labels = dbscan.fit_predict(X)
    print("Discovered clusters:", dbscan.n_clusters_)
    print("Core samples found:", len(dbscan.core_sample_indices_))

    # 3. Online streaming mini-batch clustering
    var mbk = MiniBatchKMeans(n_clusters=3, batch_size=32, random_state=42)
    mbk.partial_fit(X)
```

### Support Vector Machines (LinearSVC & LinearSVR)

```mojo
from strata import Matrix, LinearSVC, LinearSVR

def main() raises:
    # Linear Support Vector Classification via LIBLINEAR Dual Coordinate Descent
    var svc = LinearSVC(C=1.0, loss="squared_hinge", random_state=42)
    svc.fit(X_train, y_train)
    var svc_preds = svc.predict(X_test)
    var margins = svc.decision_function(X_test)

    # Linear Support Vector Regression with active-set shrinking
    var svr = LinearSVR(C=1.0, epsilon=0.1, random_state=42)
    svr.fit(X_train, y_reg)
    var svr_preds = svr.predict(X_test)
```

### Naive Bayes (MultinomialNB, BernoulliNB, ComplementNB)

```mojo
from strata import Matrix, CSRMatrix, MultinomialNB, BernoulliNB, ComplementNB

def main() raises:
    # Text count classification (supports dense Matrix and sparse CSRMatrix)
    var mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_counts, y_train)
    var mnb_probs = mnb.predict_proba(X_counts)

    # Multivariate binary features with binarization threshold
    var bnb = BernoulliNB(alpha=1.0, binarize=0.0)
    bnb.fit(X_binary, y_train)
    var bnb_preds = bnb.predict(X_binary)

    # Imbalanced text classification via complement statistics
    var cnb = ComplementNB(alpha=1.0, norm=True)
    cnb.fit(X_counts, y_train)
    var cnb_preds = cnb.predict(X_counts)
```

### Model Persistence & Serialization

```mojo
from strata import dump, load, LinearRegression, Matrix

def main() raises:
    var reg = LinearRegression(solver="cholesky")
    reg.fit(X_train, y_train)

    # Save fitted model to disk
    dump(reg, "model.strata")

    # Load model back with exact type inference and fitted state intact
    var loaded_reg = load[LinearRegression]("model.strata")
    var preds = loaded_reg.predict(X_test)
```

---

## Implemented Modules

- **`strata.core`**:
  - `Matrix[dtype]`: Dense 2D row-major matrix.
  - `MatrixView[dtype, origin]`: Zero-copy strided 2D view.
  - `CSRMatrix[dtype]`, `CSCMatrix[dtype]`: Compressed sparse row/column matrices with `spmv`, `spmm`, `spgemm`, `sddmm`.
  - `linalg`: SIMD `gemm`, `dense_dot_vec`, and LAPACK bindings (`svd`, `eigh`, `qr`, `cholesky`, `lstsq`, `solve`, `inv`).
  - `dataset`: `Dataset` container for features, targets, and metadata.
  - `interop`: NumPy and SciPy sparse conversions.
- **`strata.linear_model`**: `LinearRegression`, `Ridge`, `Lasso` (coordinate descent), `ElasticNet`, `LogisticRegression` (binary and multinomial), `SGDRegressor`, `SGDClassifier`.
- **`strata.svm`**: `LinearSVC` (L1/L2 hinge loss classification via Dual Coordinate Descent), `LinearSVR` (epsilon-insensitive loss regression with active set shrinking).
- **`strata.naive_bayes`**: `GaussianNB` (adaptive variance smoothing), `MultinomialNB` (discrete counts with dense/sparse CSR acceleration), `BernoulliNB` (multivariate binary models with binarization), `ComplementNB` (imbalanced text classification).
- **`strata.tree`**: `DecisionTreeClassifier`, `DecisionTreeRegressor` (with Gini, Entropy, MSE, MAE criteria).
- **`strata.ensemble`**: `RandomForestClassifier`, `RandomForestRegressor` (with OOB evaluation and soft voting), `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` (with UInt8 binning, histogram subtraction, and early stopping).
- **`strata.compose`**: `ColumnTransformer` (heterogeneous feature transformers applied to designated subsets of columns with configurable remainder routing).
- **`strata.io`**: `BufferWriter`, `BufferReader`, `Serializable`, `dump`, `load`, `dumps`, `loads` (zero-copy, endian-safe binary model persistence).
- **`strata.neighbors`**: `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`, `KDTree`, and distance metrics (`euclidean`, `manhattan`, `chebyshev`, `minkowski`, `cosine`, `pairwise_distances`).
- **`strata.decomposition`**: `PCA` (with whitening and sign-flip), `TruncatedSVD` (dense and sparse CSR via SpMM).
- **`strata.cluster`**: `KMeans` (k-means++, Lloyd's algorithm, distance-space transforms), `MiniBatchKMeans` (streaming online updates, EWMA inertia smoothing, `partial_fit`), `DBSCAN` (spatial density clustering, core sample indexing, non-parametric noise rejection).
- **`strata.preprocessing`**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `SimpleImputer`, `Binarizer`, `PolynomialFeatures`.
- **`strata.model_selection`**: `train_test_split`, `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `ShuffleSplit`, `StratifiedShuffleSplit`, `cross_val_score`, `cross_val_predict`, `cross_validate`, `GridSearchRegressor`, `GridSearchClassifier`, `RandomizedSearchRegressor`, `RandomizedSearchClassifier`.
- **`strata.metrics`**:
  - Regression: `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`.
  - Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_auc_score`, `log_loss`.
  - Clustering: `silhouette_score`.
- **`strata.base`**: `Transformer`, `Regressor`, `Classifier`, `Clusterer`, and composable `Pipeline` structs.

---

## Development

```bash
# Run test suite
pixi run test-all

# Format code
pixi run format

# Compile package
pixi run build
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, coding conventions, and the project roadmap.

---

*The project logo was generated using AI assistance. The software code is licensed under the MIT License, but the logo artwork is distributed as-is without copyright protection.*
