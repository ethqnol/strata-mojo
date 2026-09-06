# `StandardScaler`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Serializable, Transformer`
**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)

```mojo
struct StandardScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Serializable, Transformer)
```

```mojo
from strata.preprocessing import StandardScaler
```

**Standardize features by removing the mean and scaling to unit variance.**

The standard score of a sample $x$ is calculated as:
$$
z = \frac{x - \mu}{\sigma}
$$
where $\mu$ is the mean of the training samples and $\sigma$ is the standard deviation.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, with_mean: Bool = True, with_std: Bool = True)
```

Initialize the StandardScaler.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`with_mean`** | `Bool` | Whether to center data by subtracting feature means. Default True. |
| **`with_std`** | `Bool` | Whether to scale data to unit variance. Default True. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`mean_`** | Mean value for each feature in the training set. |
| **`scale_`** | Per-feature standard deviation scaling factor. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`StandardScaler.fit()`](#fit) | — |
| [`StandardScaler.transform()`](#transform) | — |
| [`StandardScaler.fit_transform()`](#fit_transform) | — |
| [`StandardScaler.serialize()`](#serialize) | Serializes StandardScaler parameters and fitted state into BufferWriter. |
| [`StandardScaler.deserialize()`](#deserialize) | Deserializes StandardScaler from BufferReader. |

---

## Method Details

### `StandardScaler.fit()`

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

### `StandardScaler.transform()`

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

### `StandardScaler.fit_transform()`

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

### `StandardScaler.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes StandardScaler parameters and fitted state into BufferWriter.

---

### `StandardScaler.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes StandardScaler from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.preprocessing import StandardScaler
from strata.core import Matrix

var scaler = StandardScaler[DType.float64]()
scaler.fit(X_train)
var X_scaled = scaler.transform(X_train)
```
