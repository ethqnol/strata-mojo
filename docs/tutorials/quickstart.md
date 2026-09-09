# Getting Started with Strata

Strata is a native machine learning and linear algebra library for Mojo. It provides scikit-learn compatible estimator APIs, dense and sparse matrix formats, and SIMD and LAPACK acceleration.

---

## 1. Installation

### Adding to an Existing Pixi Project

Install Strata from the official Modular Community channel on Prefix.dev:

```bash
pixi add strata --channel https://repo.prefix.dev/modular-community
```

Or add it directly to `pixi.toml`:

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
> **Linking LAPACK & BLAS**: `pixi add strata` automatically installs `liblapack` and `libblas` into `$CONDA_PREFIX/lib/`. When calling LAPACK/BLAS-accelerated routines (`svd`, `qr`, `inv`, `eigh`, `cholesky`, `solve`, `lstsq`, `gemm`, `PCA`, `TruncatedSVD`, `LinearRegression`, `Ridge`, `LogisticRegression`, or `NearestNeighbors`), pass `-Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack -Xlinker -lblas` to `mojo run`, or define a task in `pixi.toml` as shown. See [Section 2](#2-execution-and-lapackblas-configuration) for complete details and the algorithm compatibility matrix.


You can also develop Strata locally:
```bash
git clone https://github.com/ethqnol/strata-mojo.git
cd Strata
pixi install
```

---

## 2. Execution and LAPACK/BLAS Configuration

Strata routines fall into two execution profiles:

1. **Pure Mojo Routines (Zero External Dependencies)**: Decision Trees, Random Forests, Gradient Boosting, SGD, Coordinate Descent (Lasso/ElasticNet), Logistic Regression, KMeans, MiniBatchKMeans, Nearest Neighbors, Scalers (Standard/MinMax/Robust), Encoders, ColumnTransformer, and Serialization. Run directly with `mojo run main.mojo`.
2. **LAPACK & BLAS Accelerated Solvers**: Matrix decompositions (`svd`, `qr`, `inv`, `eigh`, `cholesky`, `solve`, `solve_cholesky`, `lstsq`), `PCA`, `TruncatedSVD`, `LinearRegression`, and `Ridge`. Also provides hardware-accelerated BLAS matrix multiplication (`cblas_dgemm`/`cblas_sgemm`). Require linker flags pointing to Pixi's `liblapack.so` and `libblas.so`.

### Recommended: Set Up a Pixi Task

To avoid passing linker flags manually on every run, add a `start` task to your project's `pixi.toml`:

```toml
[tasks]
start = "mojo run -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack -Xlinker -lblas main.mojo"
```

Run your application with:

```bash
pixi run start
```

### Algorithm Compatibility and Execution Matrix

| Category | Algorithm / Routine | Requires Linker Flags (`-llapack -lblas`) | Execution Command |
| :--- | :--- | :---: | :--- |
| **Ensemble & Trees** | `RandomForestClassifier`, `RandomForestRegressor` | No (Pure Mojo) | `mojo run main.mojo` |
| | `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` | No (Pure Mojo) | `mojo run main.mojo` |
| | `DecisionTreeClassifier`, `DecisionTreeRegressor` | No (Pure Mojo) | `mojo run main.mojo` |
| **Linear Models** | `Lasso`, `ElasticNet` (Coordinate Descent) | No (Pure Mojo) | `mojo run main.mojo` |
| | `SGDClassifier`, `SGDRegressor` | No (Pure Mojo) | `mojo run main.mojo` |
| | `LinearRegression` (all analytical solvers: `cholesky`, `qr`, `lstsq`, `solve`) | **Yes** (LAPACK) | `pixi run start` |
| | `Ridge` (all analytical solvers: `cholesky`, `svd`, `lsqr`, `solve`) | **Yes** (LAPACK) | `pixi run start` |
| | `LogisticRegression` (hardware GEMM gradient & inference engine) | **Yes** (BLAS) | `pixi run start` |
| **Decomposition** | `PCA`, `TruncatedSVD` (SVD eigensolvers) | **Yes** (LAPACK) | `pixi run start` |
| **Clustering** | `KMeans`, `MiniBatchKMeans` | No (Pure Mojo) | `mojo run main.mojo` |
| **Nearest Neighbors**| `KDTree` (Spatial tree indexing) | No (Pure Mojo) | `mojo run main.mojo` |
| | `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor` (GEMM distance matrix) | **Yes** (BLAS) | `pixi run start` |
| **Preprocessing** | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Binarizer` | No (Pure Mojo) | `mojo run main.mojo` |
| | `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `SimpleImputer`, `Normalizer` | No (Pure Mojo) | `mojo run main.mojo` |
| | `PolynomialFeatures`, `ColumnTransformer`, `Pipeline` | No (Pure Mojo) | `mojo run main.mojo` |
| **Model Selection** | `KFold`, `StratifiedKFold`, `ShuffleSplit`, `train_test_split` | No (Pure Mojo) | `mojo run main.mojo` |
| **Persistence** | `dump`, `load`, `dumps`, `loads` (Zero-copy binary serializer) | No (Pure Mojo) | `mojo run main.mojo` |
| **Linear Algebra** | Vector Ops, Dense Dot Product (`dense_dot_vec`), Sparse (`CSRMatrix`/`CSCMatrix`/`SpMV`/`SpMM`) | No (Pure Mojo) | `mojo run main.mojo` |
| | `svd`, `qr`, `inv`, `eigh`, `cholesky`, `solve`, `solve_cholesky`, `lstsq`, `gemm` | **Yes** (LAPACK/BLAS) | `pixi run start` |

---

## 3. Performance Benchmarks vs. Scikit-Learn

Strata provides significant throughput and latency advantages over Scikit-Learn while maintaining **100% mathematical and numerical parity**:

| Domain | Estimator / Routine | Phase | Strata Median | Scikit-Learn Median | Speedup | Parity / Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Preprocessing** | `PolynomialFeatures` | `fit` | **82.4 µs** | 602.4 µs | **7.31x faster** | Exact Match |
| | `MinMaxScaler` | `fit` | **151.7 µs** | 796.7 µs | **5.25x faster** | Exact Match |
| | `StandardScaler` | `fit` | **254.6 µs** | 1.16 ms | **4.56x faster** | Exact Match |
| | `RobustScaler` | `transform` | **129.5 µs** | 506.2 µs | **3.91x faster** | Exact Match |
| **Neighbors** | `NearestNeighbors` | `fit` | **120.9 µs** | 516.7 µs | **4.27x faster** | Exact Match |
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

*For full multi-scale benchmarks, memory profiles, and throughput tables, see the comprehensive [Performance Benchmarks](../explanation/benchmarks.md) guide.*

---

## 4. Code Examples

### Example A: Tree-Based Model (Pure Mojo)

Create `rf_example.mojo`:

```mojo
from strata import Matrix, RandomForestClassifier, accuracy_score

def main() raises:
    var X = Matrix[DType.float64](8, 2)
    X[0, 0] = -3.0; X[0, 1] = -2.0
    X[1, 0] = -2.0; X[1, 1] = -3.0
    X[2, 0] = -4.0; X[2, 1] = -2.5
    X[3, 0] = -2.5; X[3, 1] = -4.0
    X[4, 0] =  3.0; X[4, 1] =  2.0
    X[5, 0] =  2.0; X[5, 1] =  3.0
    X[6, 0] =  4.0; X[6, 1] =  2.5
    X[7, 0] =  2.5; X[7, 1] =  4.0

    var y: List[Scalar[DType.int32]] = [0, 0, 0, 0, 1, 1, 1, 1]

    var rf = RandomForestClassifier[DType.float64](n_estimators=20, max_depth=4, random_state=42)
    rf.fit(X, y)

    var preds = rf.predict(X)
    print("Training Accuracy:", accuracy_score(y, preds))
```

Run directly:
```bash
mojo run rf_example.mojo
```

---

### Example B: Linear Model & PCA (LAPACK Solvers)

Create `lapack_example.mojo`:

```mojo
from strata import Matrix, LinearRegression, PCA, dump, load

def main() raises:
    var X = Matrix[DType.float64](4, 2)
    X[0, 0] = 1.0; X[0, 1] = 1.0
    X[1, 0] = 1.0; X[1, 1] = 2.0
    X[2, 0] = 2.0; X[2, 1] = 2.0
    X[3, 0] = 2.0; X[3, 1] = 3.0
    var y: List[Scalar[DType.float64]] = [6.0, 8.0, 9.0, 11.0]

    # 1. Cholesky-accelerated Linear Regression
    var reg = LinearRegression(solver="cholesky")
    reg.fit(X, y)
    print("Fitted Intercept:", reg.intercept_)
    print("Fitted Coefficients:", reg.coef_[0], reg.coef_[1])

    # 2. SVD-accelerated Principal Component Analysis
    var pca = PCA(n_components=1)
    var X_trans = pca.fit_transform(X)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_[0])

    # 3. Model Serialization
    dump(reg, "model.strata")
    var restored = load[LinearRegression[]]("model.strata")
    var preds = restored.predict(X)
    print("Restored Model Predictions:", preds[0], preds[1], preds[2], preds[3])
```

Run with linker flags:
```bash
pixi run mojo -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack lapack_example.mojo
```
Or via your Pixi task:
```bash
pixi run start
```

---

## 4. Key Concepts

- **`Matrix[dtype]`**: Contiguous 2D dense matrix format with native SIMD layouts.
- **`fit(X, y)` and `predict(X)`**: Standard estimator lifecycle methods across all models.
- **`ColumnTransformer` & `Pipeline`**: Composable preprocessor chains with compile-time type safety.
- **`dump` / `load`**: Zero-copy binary serialization format.

For multi-stage preprocessing pipelines, see [Composing End-to-End ML Pipelines](end_to_end_pipeline.md).
