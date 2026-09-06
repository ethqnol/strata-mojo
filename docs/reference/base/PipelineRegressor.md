# `PipelineRegressor`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`
**Source**: [`strata/base/pipeline.mojo`](file:////home/ewu/Code/Strata/strata/base/pipeline.mojo)

```mojo
struct PipelineRegressor[T: Transformer, R: Regressor, target_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.base import PipelineRegressor
```

**Sequentially applies a transformer pipeline before fitting a regressor.**

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`T`** | Type of the feature transformer step. |
| **`R`** | Type of the regressor step. |
| **`target_dtype`** | Data type of target values. Default DType.float64. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`transformer`** | Transformer step instance. |
| **`regressor`** | Regressor step instance. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`PipelineRegressor.fit()`](#fit) | Fits the transformer on X, then fits the regressor on transformed features. |
| [`PipelineRegressor.predict()`](#predict) | Transforms input features and computes regression predictions. |

---

## Method Details

### `PipelineRegressor.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the transformer on X, then fits the regressor on transformed features.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `PipelineRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Transforms input features and computes regression predictions.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target values.

---
