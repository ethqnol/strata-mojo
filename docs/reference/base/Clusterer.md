# `Clusterer`

**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`
**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)

```mojo
trait Clusterer(Estimator)
```

```mojo
from strata.base import Clusterer
```

**Interface for unsupervised clustering algorithms.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Clusterer.fit()`](#fit) | — |
| [`Clusterer.predict()`](#predict) | — |
| [`Clusterer.fit_predict()`](#fit_predict) | — |
| [`Clusterer.predict_proba()`](#predict_proba) | Predicts class probabilities for a Dataset container. |
| [`Clusterer.transform()`](#transform) | Transforms dataset records and returns a new Dataset preserving labels and names. |
| [`Clusterer.fit_transform()`](#fit_transform) | Fits transformer and transforms dataset records in place. |

---

## Method Details

### `Clusterer.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[R: Regressor, feat_dtype: DType, target_dtype: DType](mut model: R, dataset: Dataset[feat_dtype, target_dtype])
def fit[C: Classifier, feat_dtype: DType, target_dtype: DType](mut model: C, dataset: Dataset[feat_dtype, target_dtype])
def fit[T: Transformer, feat_dtype: DType, target_dtype: DType](mut model: T, dataset: Dataset[feat_dtype, target_dtype])
def fit[K: Clusterer, feat_dtype: DType, target_dtype: DType](mut model: K, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Fits a Regressor using a Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

---

### `Clusterer.predict()`

```mojo
def predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]
def predict[R: Regressor, feat_dtype: DType, target_dtype: DType](model: R, dataset: Dataset[feat_dtype, target_dtype]) -> List[ Scalar[feat_dtype] ]
def predict[C: Classifier, feat_dtype: DType, target_dtype: DType](model: C, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
def predict[K: Clusterer, feat_dtype: DType, target_dtype: DType](model: K, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Predicts regression targets for a Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`model`** | `R` | — |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

**Returns**: `List[Int]`

---

### `Clusterer.fit_predict()`

```mojo
def fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `List[Int]`

---

### `Clusterer.predict_proba()`

```mojo
def predict_proba[C: Classifier, feat_dtype: DType, target_dtype: DType](model: C, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[ feat_dtype ]
```
Predicts class probabilities for a Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`model`** | `C` | — |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

**Returns**: `Matrix[ feat_dtype ]`

---

### `Clusterer.transform()`

```mojo
def transform[T: Transformer, feat_dtype: DType, target_dtype: DType](model: T, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```
Transforms dataset records and returns a new Dataset preserving labels and names.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`model`** | `T` | — |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

**Returns**: `Dataset[ feat_dtype, target_dtype ]`

---

### `Clusterer.fit_transform()`

```mojo
def fit_transform[T: Transformer, feat_dtype: DType, target_dtype: DType](mut model: T, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```
Fits transformer and transforms dataset records in place.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

**Returns**: `Dataset[ feat_dtype, target_dtype ]`

---
