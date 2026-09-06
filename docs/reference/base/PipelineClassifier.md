# `PipelineClassifier`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  
**Source**: [`strata/base/pipeline.mojo`](file:////home/ewu/Code/Strata/strata/base/pipeline.mojo)

```mojo
struct PipelineClassifier[T: Transformer, C: Classifier, target_dtype: DType = DType.int32](Classifier, Copyable, Movable)
```

```mojo
from strata.base import PipelineClassifier
```

**Sequentially applies a transformer pipeline before fitting a classifier.**

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`T`** | Type of the feature transformer step. |
| **`C`** | Type of the classifier step. |
| **`target_dtype`** | Data type of target labels. Default DType.int32. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`transformer`** | Transformer step instance. |
| **`classifier`** | Classifier step instance. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`PipelineClassifier.fit()`](#fit) | Fits the transformer on X, then fits the classifier on transformed features. |
| [`PipelineClassifier.predict()`](#predict) | Transforms input features and predicts discrete class labels. |
| [`PipelineClassifier.predict_proba()`](#predict_proba) | Transforms input features and predicts class probability distributions. |

---

## Method Details

### `PipelineClassifier.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the transformer on X, then fits the classifier on transformed features.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `PipelineClassifier.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Transforms input features and predicts discrete class labels.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]` — List[Int]: Predicted discrete class labels.

---

### `PipelineClassifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Transforms input features and predicts class probability distributions.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Probability distribution matrix of shape $(N, K)$.

---
