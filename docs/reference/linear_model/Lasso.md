# `Lasso`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor, Serializable`  
**Source**: [`strata/linear_model/lasso.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/lasso.mojo)

```mojo
struct Lasso[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor, Serializable)
```

```mojo
from strata.linear_model import Lasso
```

**Lasso linear model with L1 regularization.**

Minimizes the penalized least-squares objective function using coordinate descent:
$$
\min_{w, b} \frac{1}{2N} \|y - (Xw + b)\|_2^2 + \alpha \|w\|_1
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, alpha: Scalar[Self.compute_dtype] = 1.0, fit_intercept: Bool = True, max_iter: Int = 1000, tol: Scalar[Self.compute_dtype] = 1e-4, positive: Bool = False)
```

Initialize the Lasso regression estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`alpha`** | `Scalar[Self.compute_dtype]` | Regularization strength (must be non-negative). Default 1.0. |
| **`fit_intercept`** | `Bool` | Whether to fit an intercept term. Default True. |
| **`max_iter`** | `Int` | Maximum iterations for coordinate descent. Default 1000. |
| **`tol`** | `Scalar[Self.compute_dtype]` | Convergence tolerance threshold. Default 1e-4. |
| **`positive`** | `Bool` | Force non-negative coefficients. Default False. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`coef_`** | Weight vector coefficients of length $D$. |
| **`intercept_`** | Independent bias intercept term. |
| **`n_iter_`** | Actual number of coordinate descent iterations run. |
| **`is_fitted`** | Boolean flag indicating if the model has been fitted. |
| **`n_features_in_`** | Number of input features observed during fitting. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Lasso.fit()`](#fit) | Fit the Lasso linear model via coordinate descent. |
| [`Lasso.predict()`](#predict) | Predict continuous target values using the fitted Lasso model. |
| [`Lasso.serialize()`](#serialize) | Serializes Lasso parameters and fitted state into BufferWriter. |
| [`Lasso.deserialize()`](#deserialize) | Deserializes Lasso from BufferReader. |

---

## Method Details

### `Lasso.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit the Lasso linear model via coordinate descent.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `Lasso.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict continuous target values using the fitted Lasso model.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target vector of length $N$.

---

### `Lasso.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes Lasso parameters and fitted state into BufferWriter.

---

### `Lasso.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes Lasso from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.linear_model import Lasso
from strata.core import Matrix

var reg = Lasso[DType.float64](alpha=0.1, max_iter=1000)
reg.fit(X_train, y_train)
var preds = reg.predict(X_test)
```
