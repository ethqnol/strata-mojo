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
strata = ">=0.1.0"
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
| **Preprocessing** | `PolynomialFeatures` | `fit` | **82.4 µs** | 602.4 µs | **7.31x faster** | Exact Match |
| | `MinMaxScaler` | `fit` | **151.7 µs** | 796.7 µs | **5.25x faster** | Exact Match |
| | `StandardScaler` | `fit` | **254.6 µs** | 1.16 ms | **4.56x faster** | Exact Match |
| | `RobustScaler` | `transform` | **129.5 µs** | 506.2 µs | **3.91x faster** | Exact Match |
| **Nearest Neighbors**| `NearestNeighbors` | `fit` | **120.9 µs** | 516.7 µs | **4.27x faster** | Exact Match |
| | `KNeighborsClassifier` | `predict` | **4.34 ms** | 6.75 ms | **1.55x faster** | Exact Parity (`1.0000`) |
| | `KNeighborsRegressor` | `predict` | **4.31 ms** | 5.95 ms | **1.38x faster** | Exact Parity ($R^2 = 0.92$) |
| | `NearestNeighbors` | `kneighbors` | **4.21 ms** | 5.58 ms | **1.33x faster** | Exact Match |
| **Clustering** | `KMeans` | `fit` | **778.8 µs** | 2.49 ms | **3.19x faster** | Exact Inertia Match |
| | `MiniBatchKMeans` | `predict` | **135.4 µs** | 279.8 µs | **2.07x faster** | Exact Inertia Match |
| | `TruncatedSVD_CSR` | `fit` | **1.31 ms** | 3.15 ms | **2.41x faster** | Exact Match |
| **Trees & Ensembles**| `DecisionTreeClassifier` | `predict` | **53.9 µs** | 296.6 µs | **5.50x faster** | Exact Parity (`1.0000`) |
| | `DecisionTreeRegressor` | `predict` | **219.9 µs** | 1.01 ms | **4.60x faster** | Exact Parity ($R^2 = 0.98$) |
| | `DecisionTreeClassifier` | `fit` | **2.11 ms** | 4.75 ms | **2.25x faster** | Exact Parity (`1.0000`) |
| | `RandomForestRegressor` | `fit` | **113.26 ms** | 230.05 ms | **2.03x faster** | Exact Parity ($R^2 = 0.97$) |
| | `HistGradientBoostingRegressor` | `predict` | **2.81 ms** | 5.53 ms | **1.96x faster** | Exact Parity ($R^2 = 0.95$) |
| **Linear Algebra** | `Dense_Dot_Vec` | `matvec` | **81.9 µs** | 113.1 µs | **1.38x faster** | Exact Match |
| | `Dense_GEMM` | `matmul` | **1.37 ms** | 1.41 ms | **≈ parity (1.03x)**| Exact Match |

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

### Clustering

```mojo
from strata import Matrix, KMeans, MiniBatchKMeans

def main() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    # Cluster samples into K groups using Lloyd's algorithm
    var kmeans = KMeans(n_clusters=2, init="k-means++", random_state=42)
    kmeans.fit(X)

    var labels = kmeans.predict(X)
    var dists = kmeans.transform(X)

    # Online streaming mini-batch clustering
    var mbk = MiniBatchKMeans(n_clusters=2, batch_size=32, random_state=42)
    mbk.partial_fit(X)
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
- **`strata.tree`**: `DecisionTreeClassifier`, `DecisionTreeRegressor` (with Gini, Entropy, MSE, MAE criteria).
- **`strata.ensemble`**: `RandomForestClassifier`, `RandomForestRegressor` (with OOB evaluation and soft voting), `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` (with UInt8 binning, histogram subtraction, and early stopping).
- **`strata.compose`**: `ColumnTransformer` (heterogeneous feature transformers applied to designated subsets of columns with configurable remainder routing).
- **`strata.io`**: `BufferWriter`, `BufferReader`, `Serializable`, `dump`, `load`, `dumps`, `loads` (zero-copy, endian-safe binary model persistence).
- **`strata.neighbors`**: `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`, `KDTree`, and distance metrics (`euclidean`, `manhattan`, `chebyshev`, `minkowski`, `cosine`, `pairwise_distances`).
- **`strata.decomposition`**: `PCA` (with whitening and sign-flip), `TruncatedSVD` (dense and sparse CSR via SpMM).
- **`strata.cluster`**: `KMeans` (k-means++, Lloyd's algorithm, distance-space transforms), `MiniBatchKMeans` (streaming online updates, EWMA inertia smoothing, `partial_fit`).
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
