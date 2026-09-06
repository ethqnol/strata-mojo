# `PipelineTransformer`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`
**Source**: [`strata/base/pipeline.mojo`](file:////home/ewu/Code/Strata/strata/base/pipeline.mojo)

```mojo
struct PipelineTransformer[*Transformers: Transformer](Copyable, Movable, Transformer)
```

```mojo
from strata.base import PipelineTransformer
```

**Chains an arbitrary variadic sequence of data transformers into a single composite transformer.**

Transforms data sequentially across steps:
$$
X \to T_1(X) \to T_2(T_1(X)) \to \dots \to T_N(\dots)
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`Transformers`** | Variadic pack of Transformer types. |

---

## Constructors

```mojo
def __init__(out self, transformers: Tuple[*Self.Transformers])
```

Initializes a PipelineTransformer with a variadic tuple of transformer steps.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`transformers`** | `Tuple[*Self.Transformers]` | Tuple of transformer instances satisfying the Transformer trait. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`transformers`** | Variadic tuple of transformer instances. |
| **`is_fitted`** | Boolean flag indicating if transformer steps are fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`PipelineTransformer.fit()`](#fit) | Fits all transformers sequentially. |
| [`PipelineTransformer.transform()`](#transform) | Sequentially applies transformations across all pipeline steps. |
| [`PipelineTransformer.fit_transform()`](#fit_transform) | Fits all transformers and returns the final transformed output representation. |

---

## Method Details

### `PipelineTransformer.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fits all transformers sequentially.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

---

### `PipelineTransformer.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Sequentially applies transformations across all pipeline steps.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Transformed output matrix.

---

### `PipelineTransformer.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fits all transformers and returns the final transformed output representation.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Final transformed output matrix.
---

## Example

```mojo
from strata import StandardScaler, RobustScaler, PCA, PipelineTransformer

var pipe = PipelineTransformer(
    (StandardScaler(), RobustScaler(), PCA(n_components=2))
)
```
