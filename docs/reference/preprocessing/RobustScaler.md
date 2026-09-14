# `RobustScaler`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Serializable, Transformer`  
**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)

```mojo
struct RobustScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Serializable, Transformer)
```

```mojo
from strata.preprocessing import RobustScaler
```

**Scale features using statistics that are robust to outliers.**

Centers the data on the median and scales by the Interquartile Range (IQR):
$$
x_{\text{scaled}} = \frac{x - \text{median}}{\text{IQR}}
$$
where $\text{IQR} = Q_3 - Q_1$ (by default 75th percentile minus 25th percentile).

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, with_centering: Bool = True, with_scaling: Bool = True, quantile_min: Float64 = 25.0, quantile_max: Float64 = 75.0)
```

Initialize the RobustScaler.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`with_centering`** | `Bool` | Whether to center data by subtracting the median. Default True. |
| **`with_scaling`** | `Bool` | Whether to scale data to the quantile range. Default True. |
| **`quantile_min`** | `Float64` | Lower quantile percentage of the scaling range. Default 25.0. |
| **`quantile_max`** | `Float64` | Upper quantile percentage of the scaling range. Default 75.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`center_`** | Median value for each feature in the training set. |
| **`scale_`** | Interquartile range scaling factor for each feature. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`RobustScaler.fit()`](#fit) | — |
| [`RobustScaler.transform()`](#transform) | — |
| [`RobustScaler.fit_transform()`](#fit_transform) | — |
| [`RobustScaler.inverse_transform()`](#inverse_transform) | Undoes the centering and scaling of X. |
| [`RobustScaler.serialize()`](#serialize) | Serializes RobustScaler parameters and fitted state into BufferWriter. |
| [`RobustScaler.deserialize()`](#deserialize) | Deserializes RobustScaler from BufferReader. |

---

## Method Details

### `RobustScaler.fit()`

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

### `RobustScaler.transform()`

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

### `RobustScaler.fit_transform()`

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

### `RobustScaler.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
Undoes the centering and scaling of X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `RobustScaler.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes RobustScaler parameters and fitted state into BufferWriter.

---

### `RobustScaler.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes RobustScaler from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.preprocessing import RobustScaler
from strata.core import Matrix

var scaler = RobustScaler[DType.float64]()
scaler.fit(X_train)
var X_scaled = scaler.transform(X_train)
```
