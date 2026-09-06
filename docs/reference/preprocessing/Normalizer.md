# `Normalizer`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`
**Source**: [`strata/preprocessing/normalizer.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/normalizer.mojo)

```mojo
struct Normalizer[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import Normalizer
```

**Normalize samples individually to unit norm.**

Each sample (i.e. each row of the data matrix) with at least one non-zero
component is rescaled independently of other samples so that its norm
($L_1$, $L_2$ or $\text{max}$) equals one.
$$
x_{\text{norm}} = \frac{x}{\|x\|_p}
$$
where $\|x\|_p$ is the chosen vector norm:
- $L_1$: $\|x\|_1 = \sum_j |x_j|$
- $L_2$: $\|x\|_2 = \sqrt{\sum_j x_j^2}$
- $\text{max}$: $\|x\|_\infty = \max_j |x_j|$
Rows of all zeros remain all zeros.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, norm: String = "l2")
```

Initialize the Normalizer.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`norm`** | `String` | The norm to normalize each sample with ('l1', 'l2', or 'max'). Default 'l2'. |

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
| [`Normalizer.fit()`](#fit) | Fit the transformer on feature matrix X. |
| [`Normalizer.transform()`](#transform) | Scale each non-zero sample in X to unit norm. |
| [`Normalizer.fit_transform()`](#fit_transform) | Fit to data, then transform it. |

---

## Method Details

### `Normalizer.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fit the transformer on feature matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

---

### `Normalizer.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Scale each non-zero sample in X to unit norm.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Normalized feature matrix.

---

### `Normalizer.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fit to data, then transform it.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Normalized feature matrix.
---

## Example

```mojo
from strata.preprocessing import Normalizer
from strata.core import Matrix

var normalizer = Normalizer[DType.float64](norm="l2")
normalizer.fit(X_train)
var X_norm = normalizer.transform(X_train)
```
