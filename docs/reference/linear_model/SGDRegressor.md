# `SGDRegressor`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`
**Source**: [`strata/linear_model/sgd_regressor.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/sgd_regressor.mojo)

```mojo
struct SGDRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.linear_model import SGDRegressor
```

**Linear model fitted by minimizing a regularized empirical loss with SGD.**

Minimizes the objective function using stochastic gradient descent:
$$
\min_{w, b} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(w^T x_i + b, y_i) + \alpha \mathcal{R}(w)
$$
where $\mathcal{L}$ is the regression loss function and $\mathcal{R}$ is the penalty norm ($L_2, L_1, \text{ElasticNet}$).

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, loss: String = "squared_error", penalty: String = "l2", alpha: Scalar[Self.compute_dtype] = 1e-4, l1_ratio: Scalar[Self.compute_dtype] = 0.15, fit_intercept: Bool = True, max_iter: Int = 1000, tol: Scalar[Self.compute_dtype] = 1e-3, shuffle_data: Bool = True, epsilon: Scalar[Self.compute_dtype] = 0.1, random_state: Int = 42, learning_rate: String = "invscaling", eta0: Scalar[Self.compute_dtype] = 0.01, power_t: Scalar[Self.compute_dtype] = 0.25)
```

Initialize the SGDRegressor estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`loss`** | `String` | — |
| **`penalty`** | `String` | — |
| **`alpha`** | `Scalar[Self.compute_dtype]` | — |
| **`l1_ratio`** | `Scalar[Self.compute_dtype]` | — |
| **`fit_intercept`** | `Bool` | — |
| **`max_iter`** | `Int` | — |
| **`tol`** | `Scalar[Self.compute_dtype]` | — |
| **`shuffle_data`** | `Bool` | — |
| **`epsilon`** | `Scalar[Self.compute_dtype]` | — |
| **`random_state`** | `Int` | — |
| **`learning_rate`** | `String` | — |
| **`eta0`** | `Scalar[Self.compute_dtype]` | — |
| **`power_t`** | `Scalar[Self.compute_dtype]` | — |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`coef_`** | Weight vector coefficients of length $D$. |
| **`intercept_`** | Independent bias intercept term. |
| **`n_iter_`** | Actual number of epochs executed before convergence. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |
| **`n_features_in_`** | Number of features seen during fitting. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`SGDRegressor.fit()`](#fit) | Fit linear model with Stochastic Gradient Descent. |
| [`SGDRegressor.predict()`](#predict) | Predict continuous values using the linear SGD model. |

---

## Method Details

### `SGDRegressor.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit linear model with Stochastic Gradient Descent.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `SGDRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict continuous values using the linear SGD model.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]`
---

## Example

```mojo
from strata.linear_model import SGDRegressor
from strata.core import Matrix

var reg = SGDRegressor[DType.float64](loss="squared_error", penalty="l2", alpha=1e-4)
reg.fit(X_train, y_train)
var preds = reg.predict(X_test)
```
