# `HistGradientBoostingClassifier`

**Module**: [`strata.ensemble`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  
**Source**: [`strata/ensemble/hist_gradient_boosting_classifier.mojo`](file:////home/ewu/Code/Strata/strata/ensemble/hist_gradient_boosting_classifier.mojo)

```mojo
struct HistGradientBoostingClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)
```

```mojo
from strata.ensemble import HistGradientBoostingClassifier
```

**Histogram-based Gradient Boosting Classifier.**

Builds an ensemble of classification trees using discrete UInt8 binning,
histogram subtraction, and exact 1st/2nd order gradients for binary and
multiclass cross-entropy loss.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, loss: String = "log_loss", learning_rate: Float64 = 0.1, max_iter: Int = 100, max_leaf_nodes: Int = 31, max_depth: Int = 6, min_samples_leaf: Int = 20, l2_regularization: Float64 = 0.0, max_bins: Int = 256, min_gain_to_split: Float64 = 0.0, early_stopping: Bool = True, validation_fraction: Float64 = 0.1, n_iter_no_change: Int = 10, tol: Float64 = 1e-7, random_state: Int = 42)
```

Initializes a HistGradientBoostingClassifier with validated hyperparameters.

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
| **`loss`** | Loss function to optimize ('log_loss' or 'auto'). Default 'log_loss'. |
| **`learning_rate`** | Shrinkage multiplier for tree leaf outputs $\eta \in (0, 1]$. Default 0.1. |
| **`max_iter`** | Maximum number of boosting iterations. Default 100. |
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
| **`classes_`** | Discovered distinct class labels in sorted order. |
| **`init_raw_predictions_`** | Baseline raw margin offsets per class. |
| **`bin_thresholds_`** | Discovered continuous partition boundaries per feature. |
| **`trees_`** | Nested array of fitted trees ($T \times 1$ for binary, $T \times K$ for multiclass). |
| **`n_iter_`** | Actual number of boosting iterations completed. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |
| **`n_features_in_`** | Number of feature columns seen during fit. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`HistGradientBoostingClassifier.fit()`](#fit) | Fits the ensemble of classification trees on training data $(X, y)$. |
| [`HistGradientBoostingClassifier.decision_function()`](#decision_function) | Calculates raw margin predictions for samples in $X$. |
| [`HistGradientBoostingClassifier.predict_proba()`](#predict_proba) | Calculates calibrated class probability distributions for $X$. |
| [`HistGradientBoostingClassifier.predict()`](#predict) | Predicts class labels for samples in feature matrix $X$. |
| [`HistGradientBoostingClassifier.score()`](#score) | Returns the mean accuracy on the given test data and labels. |

---

## Method Details

### `HistGradientBoostingClassifier.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the ensemble of classification trees on training data $(X, y)$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `HistGradientBoostingClassifier.decision_function()`

```mojo
def decision_function[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
```
Calculates raw margin predictions for samples in $X$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[feat_dtype]`

---

### `HistGradientBoostingClassifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Calculates calibrated class probability distributions for $X$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]`

---

### `HistGradientBoostingClassifier.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predicts class labels for samples in feature matrix $X$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]`

---

### `HistGradientBoostingClassifier.score()`

```mojo
def score[feat_dtype: DType, target_dtype: DType](self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) -> Float64
```
Returns the mean accuracy on the given test data and labels.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |

**Returns**: `Float64`

---
