# `KNeighborsRegressor`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/neighbors/regression.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/regression.mojo)

```mojo
struct KNeighborsRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.neighbors import KNeighborsRegressor
```

**Regression based on k-nearest neighbors.**

Predicts the target value for query points by local interpolation:
- **Uniform weights**:
$$
\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i
$$
- **Distance weights**:
$$
w_i = \frac{1}{d(x, x_i)}, \quad \hat{y}(x) = \frac{\sum_{i \in N_K(x)} w_i y_i}{\sum_{i \in N_K(x)} w_i}
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision for distance arithmetic. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, n_neighbors: Int = 5, weights: String = "uniform", algorithm: String = "auto", metric: String = "euclidean", p: Float64 = 2.0)
```

Initialize KNeighborsRegressor.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_neighbors`** | `Int` | Number of nearest neighbors (>= 1). Default 5. |
| **`weights`** | `String` | Voting weight policy ('uniform', 'distance'). Default 'uniform'. |
| **`algorithm`** | `String` | Neighbor search algorithm ('auto', 'brute'). Default 'auto'. |
| **`metric`** | `String` | Distance metric identifier. Default 'euclidean'. |
| **`p`** | `Float64` | Minkowski metric exponent (>= 1.0). Default 2.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`n_samples_fit_`** | Number of samples in the fitted data. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`KNeighborsRegressor.fit()`](#fit) | Fit the k-nearest neighbors regressor from the training dataset. |
| [`KNeighborsRegressor.predict()`](#predict) | Predict the continuous target values for the provided data. |

---

## Method Details

### `KNeighborsRegressor.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit the k-nearest neighbors regressor from the training dataset.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `KNeighborsRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[ Scalar[feat_dtype] ]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict the continuous target values for the provided data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted regression values.
---

## Example

```mojo
from strata.neighbors import KNeighborsRegressor
from strata.core import Matrix

var reg = KNeighborsRegressor[DType.float64](n_neighbors=3, weights="distance")
reg.fit(X_train, y_train)
var y_pred = reg.predict(X_test)
```
