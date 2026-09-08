# `LinearSVR`

**Module**: [`strata.svm`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor, Serializable`  
**Source**: [`strata/svm/linear_svr.mojo`](file:////home/ewu/Code/Strata/strata/svm/linear_svr.mojo)

```mojo
struct LinearSVR[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor, Serializable)
```

```mojo
from strata.svm import LinearSVR
```

**Linear Support Vector Regression.**

Fits an epsilon-insensitive linear model using L2 regularization and
dual coordinate descent (LIBLINEAR algorithm, Ho & Lin 2012). Supports both
standard epsilon-insensitive loss (L1-SVR) and squared epsilon-insensitive
loss (L2-SVR).

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision for floating-point calculations. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, epsilon: Float64 = 0.0, tol: Float64 = 1e-4, C: Float64 = 1.0, loss: String = "epsilon_insensitive", fit_intercept: Bool = True, intercept_scaling: Float64 = 1.0, dual: Bool = True, max_iter: Int = 1000, random_state: Int = 42)
```

Initialize the LinearSVR estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`epsilon`** | `Float64` | Epsilon tube width (must be non-negative). Default 0.0. |
| **`tol`** | `Float64` | Convergence tolerance (must be strictly positive). Default 1e-4. |
| **`C`** | `Float64` | Regularization strength (must be strictly positive). Default 1.0. |
| **`loss`** | `String` | Loss function ('epsilon_insensitive' or 'squared_epsilon_insensitive'). Default 'epsilon_insensitive'. |
| **`fit_intercept`** | `Bool` | Whether to compute bias offset. Default True. |
| **`intercept_scaling`** | `Float64` | Synthetic feature coordinate multiplier. Default 1.0. |
| **`dual`** | `Bool` | Whether to optimize dual formulation. Default True. |
| **`max_iter`** | `Int` | Maximum epochs. Default 1000. |
| **`random_state`** | `Int` | Random shuffle seed. Default 42. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`coef_`** | Weights assigned to the features of length n_features. |
| **`intercept_`** | Constants in decision function. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`n_iter_`** | Number of iterations run until convergence. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`LinearSVR.copy()`](#copy) | Returns a deep copy of the LinearSVR instance. |
| [`LinearSVR.fit()`](#fit) | Fit the Linear Support Vector Regressor on training data. |
| [`LinearSVR.predict()`](#predict) | Predict continuous target values for feature matrix X. |
| [`LinearSVR.serialize()`](#serialize) | Serializes LinearSVR parameters and fitted state into BufferWriter. |
| [`LinearSVR.deserialize()`](#deserialize) | Deserializes LinearSVR from BufferReader. |

---

## Method Details

### `LinearSVR.copy()`

```mojo
def copy(self) -> Self
```
Returns a deep copy of the LinearSVR instance.

**Returns**: `Self`

---

### `LinearSVR.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit the Linear Support Vector Regressor on training data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `LinearSVR.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[ Scalar[feat_dtype] ]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict continuous target values for feature matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Continuous predictions of length n_samples.

---

### `LinearSVR.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes LinearSVR parameters and fitted state into BufferWriter.

---

### `LinearSVR.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes LinearSVR from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.svm import LinearSVR
from strata.core import Matrix

var X = Matrix[DType.float64](4, 2)
X[0, 0] = 1.0; X[0, 1] = 2.0
X[1, 0] = 2.0; X[1, 1] = 3.0
X[2, 0] = 3.0; X[2, 1] = 4.0
X[3, 0] = 4.0; X[3, 1] = 5.0

var y: List[Scalar[DType.float64]] = [3.0, 5.0, 7.0, 9.0]
var reg = LinearSVR(epsilon=0.0, C=1.0)
reg.fit(X, y)
var preds = reg.predict(X)
```
