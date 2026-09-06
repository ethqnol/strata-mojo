# `GridSearchClassifier`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`
**Source**: [`strata/model_selection/grid_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/grid_search.mojo)

```mojo
struct GridSearchClassifier[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](Classifier, Copyable, Movable)
```

```mojo
from strata.model_selection import GridSearchClassifier
```

**Exhaustive hyperparameter grid search for classification models.**

---

## Constructors

```mojo
def __init__(out self, var candidates: List[Self.ModelType], cv: Int = 5, scoring: String = "accuracy", refit: Bool = True)
```

Initializes the grid search classifier cross-validator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`candidates`** | — | List of candidate model configurations to evaluate. |
| **`cv`** | `Int` | Number of cross-validation folds. |
| **`scoring`** | `String` | Scoring metric name. |
| **`refit`** | `Bool` | Whether to refit the best model on the complete dataset. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`GridSearchClassifier.fit()`](#fit) | Runs stratified cross-validation across all candidate models and fits the best one. |
| [`GridSearchClassifier.predict()`](#predict) | Predicts class labels using the best discovered model configuration. |
| [`GridSearchClassifier.predict_proba()`](#predict_proba) | Predicts class probabilities using the best discovered model configuration. |

---

## Method Details

### `GridSearchClassifier.fit()`

```mojo
def fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Runs stratified cross-validation across all candidate models and fits the best one.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `GridSearchClassifier.predict()`

```mojo
def predict[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predicts class labels using the best discovered model configuration.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]`

---

### `GridSearchClassifier.predict_proba()`

```mojo
def predict_proba[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> Matrix[in_feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predicts class probabilities using the best discovered model configuration.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[in_feat_dtype]`

---
