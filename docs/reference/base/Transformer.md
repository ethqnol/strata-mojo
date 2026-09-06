# `Transformer`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`
**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)

```mojo
trait Transformer(Estimator)
```

```mojo
from strata.base import Transformer
```

**Interface for data preprocessing transformers.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Transformer.fit()`](#fit) | — |
| [`Transformer.transform()`](#transform) | — |
| [`Transformer.fit_transform()`](#fit_transform) | — |

---

## Method Details

### `Transformer.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

---

### `Transformer.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `Transformer.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---
