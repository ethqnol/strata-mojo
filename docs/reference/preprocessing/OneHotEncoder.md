# `OneHotEncoder`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/preprocessing/encoders.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/encoders.mojo)

```mojo
struct OneHotEncoder[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import OneHotEncoder
```

**Encode categorical features as a one-hot numeric array.**

The input to this transformer should be a 2D matrix of integer or float
categorical features. The features are encoded using a one-hot (also known as
'one-of-K' or 'dummy') encoding scheme.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, drop: String = "none", handle_unknown: String = "error")
```

Initialize the OneHotEncoder.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`drop`** | `String` | Category dropping strategy ('none', 'first', 'if_binary'). Default 'none'. |
| **`handle_unknown`** | `String` | Behavior for unseen categories ('error', 'ignore'). Default 'error'. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`categories_`** | Categories of each feature determined during fitting. |
| **`drop_idx_`** | Indices of dropped categories for each feature. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`OneHotEncoder.n_features_out()`](#n_features_out) | Number of indicator columns produced by transform. |
| [`OneHotEncoder.fit()`](#fit) | — |
| [`OneHotEncoder.transform()`](#transform) | — |
| [`OneHotEncoder.get_feature_names_out()`](#get_feature_names_out) | Output column names as '<feature>_<category>' pairs. |
| [`OneHotEncoder.fit_transform()`](#fit_transform) | — |
| [`OneHotEncoder.inverse_transform()`](#inverse_transform) | Recovers the original categorical values from a one-hot matrix. |

---

## Method Details

### `OneHotEncoder.n_features_out()`

```mojo
def n_features_out(self) -> Int
```
Number of indicator columns produced by transform.

**Returns**: `Int`

---

### `OneHotEncoder.fit()`

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

### `OneHotEncoder.transform()`

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

### `OneHotEncoder.get_feature_names_out()`

```mojo
def get_feature_names_out(self, input_features: List[String] = List[String]()) -> List[String]
```
Output column names as '<feature>_<category>' pairs.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`input_features`** | `List[String]` | — |

**Returns**: `List[String]`

---

### `OneHotEncoder.fit_transform()`

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

### `OneHotEncoder.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
Recovers the original categorical values from a one-hot matrix.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`
---

## Example

```mojo
from strata.preprocessing import OneHotEncoder
from strata.core import Matrix

var encoder = OneHotEncoder[DType.float64](drop="if_binary")
encoder.fit(X_cat)
var X_encoded = encoder.transform(X_cat)
```
