# `CrossValidateResult`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  
**Source**: [`strata/model_selection/validation.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/validation.mojo)

```mojo
struct CrossValidateResult(Movable)
```

```mojo
from strata.model_selection import CrossValidateResult
```

**Per-fold scores for one or more metrics from a cross-validation run.**

Scores are stored as parallel lists: metrics[m] names the metric whose
per-fold values are held in test_scores[m] and, when requested,
train_scores[m].

---

## Constructors

```mojo
def __init__(out self, var metrics: List[String], var test_scores: List[List[Float64]], var train_scores: List[List[Float64]])
```

Initializes a cross-validation result.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`metrics`** | — | Names of the evaluated metrics. |
| **`test_scores`** | — | Validation-fold scores for each metric. |
| **`train_scores`** | — | Training-fold scores for each metric, or an empty list when training scores were not requested. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`CrossValidateResult.metric_index()`](#metric_index) | Returns the position of a metric name within this result. |
| [`CrossValidateResult.test_scores_for()`](#test_scores_for) | Returns the per-fold validation scores for a named metric. |
| [`CrossValidateResult.train_scores_for()`](#train_scores_for) | Returns the per-fold training scores for a named metric. |
| [`CrossValidateResult.cross_validate()`](#cross_validate) | Evaluates several regression metrics across K folds in one pass. |

---

## Method Details

### `CrossValidateResult.metric_index()`

```mojo
def metric_index(self, metric: String) -> Int
```
Returns the position of a metric name within this result.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`metric`** | `String` | — |

**Returns**: `Int` — Index into the metrics, test_scores, and train_scores lists.

---

### `CrossValidateResult.test_scores_for()`

```mojo
def test_scores_for(self, metric: String) -> List[Float64]
```
Returns the per-fold validation scores for a named metric.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`metric`** | `String` | — |

**Returns**: `List[Float64]` — One score per fold, in fold order.

---

### `CrossValidateResult.train_scores_for()`

```mojo
def train_scores_for(self, metric: String) -> List[Float64]
```
Returns the per-fold training scores for a named metric.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`metric`** | `String` | — |

**Returns**: `List[Float64]` — One score per fold, in fold order.

---

### `CrossValidateResult.cross_validate()`

```mojo
def cross_validate[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], scoring: List[String], cv: Int = 5, return_train_score: Bool = False) -> CrossValidateResult
def cross_validate[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], splits: List[Split], scoring: List[String], return_train_score: Bool = False) -> CrossValidateResult
def cross_validate[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], scoring: List[String], cv: Int = 5, return_train_score: Bool = False) -> CrossValidateResult
def cross_validate[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], splits: List[Split], scoring: List[String], return_train_score: Bool = False) -> CrossValidateResult
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Evaluates several regression metrics across K folds in one pass.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`estimator`** | `ModelType` | — |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`scoring`** | `List[String]` | — |
| **`cv`** | `Int` | — |
| **`return_train_score`** | `Bool` | — |
| **`splits`** | `List[Split]` | — |

**Returns**: `CrossValidateResult` — Per-fold scores for every requested metric.

---
