# `ColumnTransformer`

**Module**: [`strata.compose`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`
**Source**: [`strata/compose/column_transformer.mojo`](file:////home/ewu/Code/Strata/strata/compose/column_transformer.mojo)

```mojo
struct ColumnTransformer[*Transformers: Transformer](Copyable, Movable, Transformer)
```

```mojo
from strata.compose import ColumnTransformer
```

**Applies a heterogeneous variadic sequence of transformers to column subsets of a feature matrix.**

Transforms designated column subsets independently using compile-time unrolled variadic
tuples and horizontally stacks the output representations with configurable remainder handling.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`Transformers`** | Variadic pack of Transformer types. |

---

## Constructors

```mojo
def __init__(out self, transformers: Tuple[*Self.Transformers], column_specs: List[List[Int]], remainder: String = "drop")
```

Initializes a ColumnTransformer with variadic transformers and column specs.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`transformers`** | `Tuple[*Self.Transformers]` | Tuple of transformer instances satisfying the Transformer trait. |
| **`column_specs`** | `List[List[Int]]` | List of column index lists matching the count of transformers. |
| **`remainder`** | `String` | Strategy for unselected columns ('drop' or 'passthrough'). Default 'drop'. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`transformers`** | Variadic tuple of transformer instances. |
| **`column_specs`** | List of column index lists corresponding to each transformer step. |
| **`remainder`** | Strategy for unselected columns ('drop' or 'passthrough'). Default 'drop'. Fitted Attributes: |
| **`remainder_cols_`** | Discovered unselected column indices when remainder='passthrough'. |
| **`n_features_in_`** | Total number of feature columns seen during fit. |
| **`is_fitted`** | Boolean flag indicating if transformer has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`ColumnTransformer.fit()`](#fit) | Fits all transformers on their respective designated column subsets. |
| [`ColumnTransformer.transform()`](#transform) | Transforms column subsets and horizontally stacks output representations. |
| [`ColumnTransformer.fit_transform()`](#fit_transform) | Fits all transformers and returns concatenated transformed output. |

---

## Method Details

### `ColumnTransformer.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fits all transformers on their respective designated column subsets.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

---

### `ColumnTransformer.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Transforms column subsets and horizontally stacks output representations.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Concatenated transformed feature matrix.

---

### `ColumnTransformer.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fits all transformers and returns concatenated transformed output.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Concatenated transformed feature matrix.
---

## Example

```mojo
from strata import Matrix, StandardScaler, OneHotEncoder, ColumnTransformer

var ct = ColumnTransformer(
    (StandardScaler(), OneHotEncoder()),
    List[List[Int]]([0, 1], [2]),
    remainder="passthrough",
)
```
