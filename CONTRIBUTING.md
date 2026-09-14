# Contributing to Strata

Guidelines for setting up your development environment, running tests, following coding standards, and contributing to Strata.

---

## 1. Environment Setup

Strata uses [Pixi](https://pixi.sh/) to manage dependencies and compiler toolchains.

### Linux / macOS

```bash
git clone https://github.com/ethqnol/strata-mojo.git
cd Strata
pixi install
```

### Windows (WSL 2)

Mojo runs natively on Linux and macOS. On Windows, use WSL 2 with Ubuntu:

```powershell
# In PowerShell (admin)
wsl --install
```

Inside Ubuntu:
```bash
sudo apt update && sudo apt install -y curl git build-essential
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

git clone https://github.com/ethqnol/strata-mojo.git
cd Strata
pixi install
```

---

## 2. Development Workflow

### Formatting

Strata adheres to `mojo format`. Run the formatter before committing:

```bash
pixi run format         # Auto-format all source and test files
pixi run format-check   # Check formatting without modifying files
```

### Running Tests

Tests live in `tests/` and can be executed via Pixi tasks:

```bash
# Run the complete test suite
pixi run test-all

# Run specific test modules
pixi run test-matrix              # Matrix containers, GEMM, dot_vec, slicing
pixi run test-linalg              # LAPACK factorizations (SVD, QR, Cholesky, Solve)
pixi run test-sparse              # CSR/CSC matrices and sparse linear algebra
pixi run test-pipelines           # Estimator and Transformer pipelines
pixi run test-preprocessing       # Scalers, encoders, imputers
pixi run test-tree                # Decision trees and splitters
pixi run test-ensemble            # Random forests and HistGradientBoosting
pixi run test-linear-regression   # Analytical linear regression solvers
pixi run test-logistic-regression # Logistic regression
pixi run test-metrics             # Classification, regression, and clustering metrics
pixi run test-model-selection     # KFold, TimeSeriesSplit, cross_val_score, search
```

### Building & Precompiling

Verify package compilation and trait conformance:

```bash
pixi run build          # Precompiles strata into strata.mojoc
```

---

## 3. Codebase Layout

```
Strata/
├── strata/
│   ├── base/             # Estimator, Transformer, Regressor, Classifier, Pipeline traits
│   ├── core/             # Matrix, MatrixView, CSRMatrix, CSCMatrix, linalg, sparse_ops
│   ├── exceptions/       # Domain errors (DimensionMismatchError, NotFittedError, etc.)
│   ├── utils/            # Validation helpers, numerically stable math, PRNG
│   ├── preprocessing/    # Scalers (Standard, MinMax, Robust), Encoders, Imputers
│   ├── model_selection/  # Cross-validation splitters, grid search, random search
│   ├── metrics/          # Scoring metrics (regression, classification, clustering)
│   ├── linear_model/     # LinearRegression, Ridge, Lasso, ElasticNet, Logistic, SGD
│   ├── cluster/          # KMeans, MiniBatchKMeans
│   ├── tree/             # DecisionTreeClassifier, DecisionTreeRegressor
│   ├── ensemble/         # RandomForest, HistGradientBoosting
│   ├── decomposition/    # PCA, TruncatedSVD
│   ├── neighbors/        # NearestNeighbors, KDTree, KNeighbors
│   ├── compose/          # ColumnTransformer
│   └── io/               # Binary serializer / model persistence
└── tests/                # Native Mojo unit and integration test suites
```

---

## 4. Coding Standards

### Memory and Ownership
* **Non-owning views**: Use `MatrixView` for strided reads and sub-matrix slices to avoid allocations.
* **Ownership transfers**: Use `^` (move operator) when transferring ownership of arrays or fitted state into struct fields.
* **Explicit copies**: Avoid hidden allocations. Call `.copy()` explicitly when a duplicate buffer is needed.

### Data Types and Precision
* **Linear Algebra**: Low-level kernels (`gemm`, `spmm`, `dense_dot_vec`) are parameterized by `DType` and operate homogeneously.
* **Estimators**: Estimators take a compile-time `compute_dtype: DType = DType.float64`. Cast inputs at entry points via `X.cast[Self.compute_dtype]()` when required.

### Trait Contracts
Estimators implement base traits from `strata.base`:
* `Transformer`: `fit(X)`, `transform(X) -> Matrix`, `fit_transform(X) -> Matrix`
* `Regressor`: `fit(X, y)`, `predict(X) -> List[Scalar]`
* `Classifier`: `fit(X, y)`, `predict(X) -> List[Scalar]`, `predict_proba(X) -> Matrix`
* `Clusterer`: `fit(X)`, `predict(X) -> List[Int]`, `fit_predict(X) -> List[Int]`

### Input Validation & Errors
* Validate inputs at the start of `fit` and `transform` using `check_array`, `check_X_y`, or `check_sparse`.
* Verify fitted status in inference methods using `check_is_fitted("EstimatorName", self.is_fitted)`.
* Raise domain-specific exceptions from `strata.exceptions` (`DimensionMismatchError`, `NotFittedError`, `InvalidParameterError`).

---

## 5. Docstring Guidelines

All public structs, traits, constructors, and functions must include docstrings conforming to the format parsed by `scripts/generate_docs.py`.

### Rules
1. **Summary line**: First line must be a concise sentence (max 80 chars) in the imperative mood (`"Compute mean squared error..."`).
2. **Spacing**: Follow the summary with a single blank line before descriptions or sections.
3. **Sections**: Use standard Google-style section headers:
   * `Parameters:` Compile-time struct parameters (`[compute_dtype: DType]`).
   * `Args:` Runtime constructor/function arguments (`n_estimators: Int`).
   * `Attributes:` Public fields populated after `fit()` (use trailing underscore: `coef_`, `classes_`).
   * `Returns:` Return type and description.
   * `Raises:` Error types and triggers.
   * `Examples:` Fenced Mojo code block (` ```mojo ... ``` `).
4. **LaTeX Math**: Use `$ ... $` for inline math (`$N \times D$`, `$\|w\|_2$`) and `$$ ... $$` for display equations. Do not use ASCII bars like `||y - Xw||`, as raw pipes break Markdown tables.

### Example

```mojo
struct LinearRegression[
    compute_dtype: DType = DType.float64,
](Regressor, Copyable, Movable):
    """Ordinary Least Squares Linear Regression.

    Fits a linear model by minimizing the residual sum of squares:

    $$
    \min_{w} \frac{1}{2N} \|y - Xw\|_2^2
    $$

    Parameters:
        compute_dtype: Computational precision type. Default DType.float64.

    Args:
        fit_intercept: Whether to calculate the intercept term. Default True.
        solver: Solver routine ('lstsq', 'qr', 'cholesky', 'solve'). Default 'lstsq'.

    Attributes:
        coef_: Weight coefficients of length D.
        intercept_: Bias intercept term.

    Raises:
        InvalidParameterError: If solver is unrecognized.

    Examples:
        ```mojo
        from strata.linear_model import LinearRegression
        from strata.core import Matrix

        var reg = LinearRegression[DType.float64](solver="cholesky")
        reg.fit(X_train, y_train)
        var preds = reg.predict(X_test)
        ```
    """
```

---

## 6. Roadmap & Priorities

Current implementation status and planned roadmap items:

### Implemented
* **Core & Linear Algebra**: `Matrix[DType]`, `MatrixView`, `CSRMatrix`, `CSCMatrix`, LAPACK bindings (SVD, QR, Cholesky, Solve, Eigh, LU), SIMD kernels.
* **Composition**: `Pipeline`, `ColumnTransformer`, zero-copy binary serialization (`strata.io`).
* **Linear Models**: `LinearRegression` (analytical OLS), `Ridge`, `Lasso` (coordinate descent), `ElasticNet`, `LogisticRegression`, `SGDClassifier`, `SGDRegressor`.
* **Support Vector Machines**: `LinearSVC` (L1/L2 hinge loss classification via Dual Coordinate Descent), `LinearSVR` (epsilon-insensitive loss regression with active set shrinking).
* **Trees & Ensembles**: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`.
* **Neighbors & Clustering**: `NearestNeighbors`, `KDTree` (k-NN and radius queries), `KNeighborsClassifier`, `KNeighborsRegressor`, `KMeans`, `MiniBatchKMeans`, `DBSCAN` (spatial density clustering with noise detection).
* **Dimensionality Reduction**: `PCA`, `TruncatedSVD` (dense and sparse).
* **Preprocessing**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `SimpleImputer`, `Binarizer`, `PolynomialFeatures`.
* **Naive Bayes**: `GaussianNB` (continuous features with adaptive variance smoothing), `MultinomialNB` (discrete counts with dense and sparse `CSRMatrix` acceleration), `BernoulliNB` (multivariate binary models with binarization), `ComplementNB` (imbalanced text classification).
* **Model Selection & Metrics**: `train_test_split`, `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `ShuffleSplit`, `StratifiedShuffleSplit`, `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `cross_val_predict`, regression/classification/clustering metrics.

### Next Priorities
1. **Multi-Threading / Parallelization**:
   - Parallel tree training for `RandomForest` via worker threads.
   - Parallel fold evaluation in `GridSearchCV` / `cross_val_score`.
   - Threaded batch distance calculations in `NearestNeighbors`.
2. **Clustering & Spatial**:
   - `HDBSCAN` (hierarchical density-based clustering).
   - `OPTICS` (ordering points to identify clustering structure).
   - `KModes` (categorical clustering).
3. **Kernel Support Vector Machines**:
   - Non-linear `SVC` / `SVR` with RBF, polynomial, and sigmoid kernels via SMO algorithm.
4. **Probabilistic Models**:
   - `CategoricalNB` (discrete categorical features).
   - `GaussianMixture` (EM algorithm with spherical, diag, full covariance).
5. **Meta-Estimators**:
   - `VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor`.
   - `BaggingClassifier`, `BaggingRegressor`.
6. **Ecosystem & Interop**:
   - Python bindings for direct drop-in use in Python workflows.
   - ONNX export for tree and linear models.

---

## 7. Pre-PR Checklist

Run through these checks before submitting a PR:

1. **Format code**:
   ```bash
   pixi run format
   ```
2. **Compile package**:
   ```bash
   pixi run build
   ```
   Ensure zero compiler errors and warnings.
3. **Run unit tests**:
   ```bash
   pixi run test-all
   ```
   Ensure 100% of tests pass.
4. **Documentation**:
   ```bash
   pixi run generate-docs
   ```
   Ensure docstrings follow Section 5 and generated docs build without issues.
5. **Git branch**:
   Work on a topic branch (`feat/feature-name`) and open PR against `main`.
