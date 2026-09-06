# `Binarizer`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`
**Source**: [`strata/preprocessing/binarizer.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/binarizer.mojo)

```mojo
struct Binarizer[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import Binarizer
```

**Binarize feature values according to a threshold.**

Values strictly greater than the threshold map to 1, while values less than
or equal to the threshold map to 0:
$$
x_{\text{bin}} = \begin{cases} 1 & \text{if } x > \text{threshold} \\ 0 & \text{otherwise} \end{cases}
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, threshold: Scalar[Self.compute_dtype] = 0.0)
```

Initialize the Binarizer.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`threshold`** | `Scalar[Self.compute_dtype]` | Feature threshold boundary. Values greater than this map to 1. Default 0.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Binarizer.fit()`](#fit) | — |
| [`Binarizer.transform()`](#transform) | — |
| [`Binarizer.fit_transform()`](#fit_transform) | — |

---

## Method Details

### `Binarizer.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

---

### `Binarizer.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]`

---

### `Binarizer.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]`
---

## Example

```mojo
from strata.preprocessing import Binarizer
from strata.core import Matrix

var binarizer = Binarizer[DType.float64](threshold=0.5)
binarizer.fit(X_train)
var X_bin = binarizer.transform(X_train)
```
