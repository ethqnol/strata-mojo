# `LinearRegression`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor, Serializable`  
**Source**: [`strata/linear_model/linear_regression.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/linear_regression.mojo)

```mojo
struct LinearRegression[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor, Serializable)
```

```mojo
from strata.linear_model import LinearRegression
```

**Ordinary Least Squares Linear Regression.**

Fits a linear model with coefficients $w = (w_1, \dots, w_D)$ and intercept $b$
to minimize the residual sum of squares between observed targets and predictions:
$$
\min_{w, b} \frac{1}{2N} \|y - (Xw + b)\|_2^2
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, fit_intercept: Bool = True, solver: String = "lstsq")
```

Initialize the linear regression estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`fit_intercept`** | `Bool` | Whether to calculate the intercept bias term. Default True. |
| **`solver`** | `String` | Solver algorithm ('lstsq', 'qr', 'cholesky', 'solve'). Default 'lstsq'. |

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
| [`LinearRegression.fit()`](#fit) | Fit the linear model from training data. |
| [`LinearRegression.predict()`](#predict) | Predict continuous target values using the fitted linear model. |
| [`LinearRegression.serialize()`](#serialize) | Serializes LinearRegression parameters and fitted state into BufferWriter. |
| [`LinearRegression.deserialize()`](#deserialize) | Deserializes LinearRegression from BufferReader. |

---

## Method Details

### `LinearRegression.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit the linear model from training data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `LinearRegression.predict()`

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

### `LinearRegression.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes LinearRegression parameters and fitted state into BufferWriter.

---

### `LinearRegression.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes LinearRegression from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.linear_model import LinearRegression
from strata.core import Matrix

var reg = LinearRegression[DType.float64](solver="cholesky")
reg.fit(X_train, y_train)
var preds = reg.predict(X_test)
```
