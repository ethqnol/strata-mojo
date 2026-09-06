# `Regressor`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`
**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)

```mojo
trait Regressor(Estimator)
```

```mojo
from strata.base import Regressor
```

**Interface for supervised regression models.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Regressor.fit()`](#fit) | — |
| [`Regressor.predict()`](#predict) | — |

---

## Method Details

### `Regressor.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |

---

### `Regressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `List[Scalar[feat_dtype]]`

---
