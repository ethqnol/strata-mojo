# `PolynomialFeatures`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/preprocessing/polynomial.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/polynomial.mojo)

```mojo
struct PolynomialFeatures[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import PolynomialFeatures
```

**Generate polynomial and interaction features.**

Generates a new feature matrix consisting of all polynomial combinations
of the features with degree less than or equal to the specified degree.
For example, if an input sample is 2D and of the form $[a, b]$, the
degree-2 polynomial features with bias are $[1, a, b, a^2, ab, b^2]$.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, degree: Int = 2, interaction_only: Bool = False, include_bias: Bool = True)
```

Initialize the PolynomialFeatures transformer.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`degree`** | `Int` | Maximum polynomial degree (>= 0). Default 2. |
| **`interaction_only`** | `Bool` | Whether to produce only interaction terms. Default False. |
| **`include_bias`** | `Bool` | Whether to include a bias column (degree 0). Default True. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`powers_`** | Exponent matrix with shape (n_output_features_, n_features_in_). |
| **`n_features_in_`** | Number of features seen during fit. |
| **`n_output_features_`** | Total number of polynomial output features. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`PolynomialFeatures.fit()`](#fit) | Compute the number of output features and combination powers. |
| [`PolynomialFeatures.transform()`](#transform) | Transform data matrix X to polynomial feature combinations. |
| [`PolynomialFeatures.fit_transform()`](#fit_transform) | Fit to data, then transform it. |

---

## Method Details

### `PolynomialFeatures.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Compute the number of output features and combination powers.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

---

### `PolynomialFeatures.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Transform data matrix X to polynomial feature combinations.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Expanded polynomial matrix with shape (n_samples, n_output_features_).

---

### `PolynomialFeatures.fit_transform()`

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

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Expanded polynomial matrix.
---

## Example

```mojo
from strata.preprocessing import PolynomialFeatures
from strata.core import Matrix

var poly = PolynomialFeatures[DType.float64](degree=2)
poly.fit(X)
var X_poly = poly.transform(X)
```
