# `MinMaxScaler`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Serializable, Transformer`
**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)

```mojo
struct MinMaxScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Serializable, Transformer)
```

```mojo
from strata.preprocessing import MinMaxScaler
```

**Transform features by scaling each feature to a specified range.**

Scales and translates each feature individually such that it is in the given
range on the training set, e.g. between zero and one:
$$
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (\text{max} - \text{min}) + \text{min}
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, feature_range_min: Scalar[Self.compute_dtype] = 0.0, feature_range_max: Scalar[Self.compute_dtype] = 1.0, clip: Bool = False)
```

Initialize the MinMaxScaler.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`feature_range_min`** | `Scalar[Self.compute_dtype]` | Lower bound of the transformed range. Default 0.0. |
| **`feature_range_max`** | `Scalar[Self.compute_dtype]` | Upper bound of the transformed range. Default 1.0. |
| **`clip`** | `Bool` | Whether to clip transformed values to the feature range. Default False. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`data_min_`** | Per-feature minimum seen in the training data. |
| **`data_max_`** | Per-feature maximum seen in the training data. |
| **`data_range_`** | Per-feature range ($x_{\max} - x_{\min}$) seen in the data. |
| **`scale_`** | Per-feature relative scaling factor. |
| **`min_`** | Per-feature minimum adjustment. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`MinMaxScaler.fit()`](#fit) | — |
| [`MinMaxScaler.transform()`](#transform) | — |
| [`MinMaxScaler.fit_transform()`](#fit_transform) | — |
| [`MinMaxScaler.inverse_transform()`](#inverse_transform) | Undoes the scaling of X according to the fitted feature range. |
| [`MinMaxScaler.serialize()`](#serialize) | Serializes MinMaxScaler parameters and fitted state into BufferWriter. |
| [`MinMaxScaler.deserialize()`](#deserialize) | Deserializes MinMaxScaler from BufferReader. |

---

## Method Details

### `MinMaxScaler.fit()`

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

### `MinMaxScaler.transform()`

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

### `MinMaxScaler.fit_transform()`

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

### `MinMaxScaler.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
Undoes the scaling of X according to the fitted feature range.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `MinMaxScaler.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes MinMaxScaler parameters and fitted state into BufferWriter.

---

### `MinMaxScaler.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes MinMaxScaler from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.preprocessing import MinMaxScaler
from strata.core import Matrix

var scaler = MinMaxScaler[DType.float64](feature_range_min=0.0, feature_range_max=1.0)
scaler.fit(X_train)
var X_scaled = scaler.transform(X_train)
```
