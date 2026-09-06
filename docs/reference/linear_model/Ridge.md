# `Ridge`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor, Serializable`  
**Source**: [`strata/linear_model/ridge.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/ridge.mojo)

```mojo
struct Ridge[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor, Serializable)
```

```mojo
from strata.linear_model import Ridge
```

**Ridge regression with L2 regularization.**

Minimizes the penalized objective function:
$$
\min_{w} \|y - Xw\|_2^2 + \alpha \|w\|_2^2
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, alpha: Scalar[Self.compute_dtype] = 1.0, fit_intercept: Bool = True, solver: String = "auto")
```

Initialize the Ridge regression estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`alpha`** | `Scalar[Self.compute_dtype]` | Regularization strength (must be non-negative). Default 1.0. |
| **`fit_intercept`** | `Bool` | Whether to calculate the intercept bias term. Default True. |
| **`solver`** | `String` | Solver algorithm ('auto', 'cholesky', 'svd', 'solve'). Default 'auto'. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`coef_`** | Weight vector coefficients of length $D$. |
| **`intercept_`** | Independent bias intercept term. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Ridge.fit()`](#fit) | Fit the Ridge regression model from training data. |
| [`Ridge.predict()`](#predict) | Predict continuous target values using the fitted linear model. |
| [`Ridge.serialize()`](#serialize) | Serializes Ridge parameters and fitted state into BufferWriter. |
| [`Ridge.deserialize()`](#deserialize) | Deserializes Ridge from BufferReader. |

---

## Method Details

### `Ridge.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit the Ridge regression model from training data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `Ridge.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict continuous target values using the fitted linear model.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target vector of length $N$.

---

### `Ridge.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes Ridge parameters and fitted state into BufferWriter.

---

### `Ridge.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes Ridge from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.linear_model import Ridge
from strata.core import Matrix

var model = Ridge[DType.float64](alpha=0.5)
model.fit(X_train, y_train)
var preds = model.predict(X_test)
```
