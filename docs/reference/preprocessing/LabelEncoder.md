# `LabelEncoder`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`
**Source**: [`strata/preprocessing/encoders.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/encoders.mojo)

```mojo
struct LabelEncoder[compute_dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.preprocessing import LabelEncoder
```

**Encode target labels with value between 0 and n_classes-1.**

Used to transform non-numerical or non-consecutive 1D target labels
into continuous integer labels for classification tasks.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision used for internal class representation. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self)
```

Initialize the LabelEncoder.

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Distinct classes seen during fit, in sorted order. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`LabelEncoder.fit()`](#fit) | Fit label encoder on target vector y. |
| [`LabelEncoder.transform()`](#transform) | Transform target labels to normalized encoding indices. |
| [`LabelEncoder.fit_transform()`](#fit_transform) | Fit label encoder and return encoded integer labels. |
| [`LabelEncoder.inverse_transform()`](#inverse_transform) | Transform integer labels back to original encoding. |

---

## Method Details

### `LabelEncoder.fit()`

```mojo
def fit[in_dtype: DType](mut self, y: List[Scalar[in_dtype]])
```
Fit label encoder on target vector y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`y`** | `List[Scalar[in_dtype]]` | Target vector / class labels. |

---

### `LabelEncoder.transform()`

```mojo
def transform[in_dtype: DType](self, y: List[Scalar[in_dtype]]) -> List[Int]
```
Transform target labels to normalized encoding indices.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`y`** | `List[Scalar[in_dtype]]` | Target vector / class labels. |

**Returns**: `List[Int]` — List[Int]: Encoded integer labels in range [0, n_classes - 1].

---

### `LabelEncoder.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, y: List[Scalar[in_dtype]]) -> List[Int]
```
Fit label encoder and return encoded integer labels.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`y`** | `List[Scalar[in_dtype]]` | Target vector / class labels. |

**Returns**: `List[Int]` — List[Int]: Encoded integer labels.

---

### `LabelEncoder.inverse_transform()`

```mojo
def inverse_transform[out_dtype: DType = Self.compute_dtype](self, y: List[Int]) -> List[Scalar[out_dtype]]
```
Transform integer labels back to original encoding.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`out_dtype`** | — | Output precision data type. Default compute_dtype. |

**Returns**: `List[Scalar[out_dtype]]` — List[Scalar[out_dtype]]: Reconstructed original labels.
---

## Example

```mojo
from strata.preprocessing import LabelEncoder

var encoder = LabelEncoder[DType.float64]()
encoder.fit(y_train)
var y_encoded = encoder.transform(y_train)
var y_original = encoder.inverse_transform(y_encoded)
```
