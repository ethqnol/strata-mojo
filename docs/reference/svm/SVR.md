# `SVR`

**Module**: [`strata.svm`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor, Serializable`  
**Source**: [`strata/svm/svr.mojo`](file:////home/ewu/Code/strata-mojo/strata/svm/svr.mojo)

```mojo
struct SVR[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor, Serializable)
```

```mojo
from strata.svm import SVR
```

**Epsilon-Support Vector Regression using non-linear kernels.**

Fits an epsilon-insensitive regression model via dual coordinate descent
across kernel mappings (RBF, Polynomial, Sigmoid, and Linear):
$$
\min_{\beta} \frac{1}{2} \beta^T Q \beta - y^T \beta + \epsilon \sum_{i=1}^n |\beta_i|
\quad \text{s.t.} \quad -C \le \beta_i \le C
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Floating point precision for kernel operations. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, C: Float64 = 1.0, epsilon: Float64 = 0.1, kernel: String = "rbf", degree: Int = 3, gamma: String = "scale", gamma_val: Float64 = -1.0, coef0: Float64 = 0.0, tol: Float64 = 1e-3, fit_intercept: Bool = True, max_iter: Int = 10000)
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`C`** | `Float64` | Regularization parameter. Must be strictly positive. Default 1.0. |
| **`epsilon`** | `Float64` | Epsilon-tube width within which no penalty is associated in the training loss function. Must be non-negative. Default 0.1. |
| **`kernel`** | `String` | Kernel routine ('rbf', 'linear', 'poly', 'sigmoid'). Default 'rbf'. |
| **`degree`** | `Int` | Degree of the polynomial kernel function. Default 3. |
| **`gamma`** | `String` | Kernel coefficient mode ('scale' or 'auto'). Default 'scale'. |
| **`gamma_val`** | `Float64` | Explicit numeric gamma. Overrides gamma mode when > 0. Default -1.0. |
| **`coef0`** | `Float64` | Independent term in poly and sigmoid kernels. Default 0.0. |
| **`tol`** | `Float64` | Tolerance for stopping criterion. Default 1e-3. |
| **`fit_intercept`** | `Bool` | Whether to estimate an intercept term. Default True. |
| **`max_iter`** | `Int` | Hard limit on solver iterations. Default 10000. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`support_vectors_`** | Support vectors of shape (n_support_vectors, n_features). |
| **`support_indices_`** | Indices of support vectors in training set. |
| **`dual_coef_`** | Dual coefficients for support vectors of length n_support_vectors. |
| **`intercept_`** | Bias term in decision function. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`n_iter_`** | Number of iterations run by the solver. |
| **`gamma_`** | Resolved kernel scale coefficient. |
| **`is_fitted`** | Whether the estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`SVR.fit()`](#fit) | Fits the kernel Support Vector Regressor on features X and targets y. |
| [`SVR.predict()`](#predict) | Predict continuous target values for samples in X. |
| [`SVR.serialize()`](#serialize) | Serializes SVR parameters and fitted state into BufferWriter. |
| [`SVR.deserialize()`](#deserialize) | Deserializes SVR from BufferReader. |

---

## Method Details

### `SVR.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the kernel Support Vector Regressor on features X and targets y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Training feature matrix of shape (n_samples, n_features). |
| **`y`** | `List[Scalar[target_dtype]]` | Target values of length n_samples. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `SVR.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[ Scalar[feat_dtype] ]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict continuous target values for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Input feature matrix of shape (n_samples, n_features). |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target values of length n_samples.

---

### `SVR.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes SVR parameters and fitted state into BufferWriter.

---

### `SVR.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes SVR from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.svm import SVR
from strata.core import Matrix

var reg = SVR[DType.float64](C=1.0, epsilon=0.1, kernel="rbf")
reg.fit(X_train, y_train)
var preds = reg.predict(X_test)
```
