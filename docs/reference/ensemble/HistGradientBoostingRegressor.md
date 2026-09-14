# `HistGradientBoostingRegressor`

**Module**: [`strata.ensemble`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/ensemble/hist_gradient_boosting_regressor.mojo`](file:////home/ewu/Code/Strata/strata/ensemble/hist_gradient_boosting_regressor.mojo)

```mojo
struct HistGradientBoostingRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.ensemble import HistGradientBoostingRegressor
```

**Histogram-based Gradient Boosting Regressor.**

Builds an ensemble of regression trees iteratively fit to negative gradients
of the squared error loss function using discrete UInt8 binning and
histogram subtraction.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, loss: String = "squared_error", learning_rate: Float64 = 0.1, max_iter: Int = 100, max_leaf_nodes: Int = 31, max_depth: Int = 6, min_samples_leaf: Int = 20, l2_regularization: Float64 = 0.0, max_bins: Int = 256, min_gain_to_split: Float64 = 0.0, early_stopping: Bool = True, validation_fraction: Float64 = 0.1, n_iter_no_change: Int = 10, tol: Float64 = 1e-7, random_state: Int = 42)
```

Initializes a HistGradientBoostingRegressor with validated hyperparameters.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`loss`** | `String` | — |
| **`learning_rate`** | `Float64` | — |
| **`max_iter`** | `Int` | — |
| **`max_leaf_nodes`** | `Int` | — |
| **`max_depth`** | `Int` | — |
| **`min_samples_leaf`** | `Int` | — |
| **`l2_regularization`** | `Float64` | — |
| **`max_bins`** | `Int` | — |
| **`min_gain_to_split`** | `Float64` | — |
| **`early_stopping`** | `Bool` | — |
| **`validation_fraction`** | `Float64` | — |
| **`n_iter_no_change`** | `Int` | — |
| **`tol`** | `Float64` | — |
| **`random_state`** | `Int` | — |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`loss`** | Loss function to optimize. Currently 'squared_error'. Default 'squared_error'. |
| **`learning_rate`** | Shrinkage multiplier for tree leaf outputs $\eta \in (0, 1]$. Default 0.1. |
| **`max_iter`** | Maximum number of boosting iterations (number of trees). Default 100. |
| **`max_leaf_nodes`** | Maximum number of leaves per tree. Default 31. |
| **`max_depth`** | Maximum tree depth. Default 6. |
| **`min_samples_leaf`** | Minimum samples required per leaf node. Default 20. |
| **`l2_regularization`** | $L_2$ leaf weight regularization parameter $\lambda \ge 0$. Default 0.0. |
| **`max_bins`** | Maximum discrete histogram bins ($2 \le \text{max\_bins} \le 256$). Default 256. |
| **`min_gain_to_split`** | Minimum gain required to split an internal node. Default 0.0. |
| **`early_stopping`** | Whether to use validation-based early stopping. Default True. |
| **`validation_fraction`** | Proportion of training data set aside for early stopping. Default 0.1. |
| **`n_iter_no_change`** | Maximum consecutive iterations with non-improving validation loss. Default 10. |
| **`tol`** | Minimum relative loss improvement threshold. Default 1e-7. |
| **`random_state`** | Seed for PRNG shuffling. Default 42. Fitted Attributes: |
| **`init_raw_prediction_`** | Baseline raw prediction offset (sample mean $\bar{y}$). |
| **`bin_thresholds_`** | Discovered continuous partition boundaries per feature. |
| **`trees_`** | List of sequential fitted HistTree models. |
| **`n_iter_`** | Actual number of boosting iterations completed. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |
| **`n_features_in_`** | Number of feature columns seen during fit. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`HistGradientBoostingRegressor.fit()`](#fit) | Fits the ensemble of histogram gradient boosted trees on training data $(X, y)$. |
| [`HistGradientBoostingRegressor.predict()`](#predict) | Predicts continuous target values for feature matrix $X$. |
| [`HistGradientBoostingRegressor.score()`](#score) | Returns the coefficient of determination $R^2$ on test data $(X, y)$. |

---

## Method Details

### `HistGradientBoostingRegressor.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the ensemble of histogram gradient boosted trees on training data $(X, y)$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `HistGradientBoostingRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predicts continuous target values for feature matrix $X$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target vector of length $N$.

---

### `HistGradientBoostingRegressor.score()`

```mojo
def score[feat_dtype: DType, target_dtype: DType](self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) -> Float64
```
Returns the coefficient of determination $R^2$ on test data $(X, y)$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |

**Returns**: `Float64`

---
