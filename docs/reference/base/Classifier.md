# `Classifier`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`  
**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)

```mojo
trait Classifier(Estimator)
```

```mojo
from strata.base import Classifier
```

**Interface for supervised classification models.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Classifier.fit()`](#fit) | — |
| [`Classifier.predict()`](#predict) | — |
| [`Classifier.predict_proba()`](#predict_proba) | — |

---

## Method Details

### `Classifier.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |

---

### `Classifier.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `List[Int]`

---

### `Classifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[feat_dtype]`

---
