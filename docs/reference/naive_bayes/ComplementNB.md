# `ComplementNB`

**Module**: [`strata.naive_bayes`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`  
**Source**: [`strata/naive_bayes/complement.mojo`](file:////home/ewu/Code/Strata/strata/naive_bayes/complement.mojo)

```mojo
struct ComplementNB[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.naive_bayes import ComplementNB
```

**Complement Naive Bayes classifier (Rennie et al., 2003).**

Designed for text classification with imbalanced class distributions.
Instead of calculating probabilities of features given class $c$,
ComplementNB calculates probabilities of features given all classes *except* $c$,
mitigating the majority class bias inherent in standard MultinomialNB.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, alpha: Scalar[Self.compute_dtype] = 1.0, fit_prior: Bool = True, class_prior: List[Scalar[Self.compute_dtype]] = List[ Scalar[Self.compute_dtype] ](), norm: Bool = False)
```

Initialize the ComplementNB classifier.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`alpha`** | `Scalar[Self.compute_dtype]` | Additive smoothing parameter (>= 0). Default 1.0. |
| **`fit_prior`** | `Bool` | Whether to learn class prior probabilities. Default True. |
| **`class_prior`** | `List[Scalar[Self.compute_dtype]]` | Prior probabilities of the classes. Default empty. |
| **`norm`** | `Bool` | Whether to normalize weights. Default False. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique class labels observed during fit. |
| **`class_count_`** | Number of training samples observed in each class. |
| **`class_log_prior_`** | Smoothed empirical log-probability of each class. |
| **`feature_count_`** | Number of samples encountered for each (class, feature) pair of shape $(K, D)$. |
| **`feature_all_`** | Total count of each feature across all classes of length $D$. |
| **`feature_log_prob_`** | Weights for decision making of shape $(K, D)$. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`ComplementNB.copy()`](#copy) | Returns a deep copy of the ComplementNB instance. |
| [`ComplementNB.fit()`](#fit) | Fit Complement Naive Bayes using dense feature matrix X and target labels y. |
| [`ComplementNB.predict_log_proba()`](#predict_log_proba) | Predict class log probabilities for dense matrix X. |
| [`ComplementNB.predict_proba()`](#predict_proba) | Predict class probabilities for dense matrix X. |
| [`ComplementNB.predict()`](#predict) | Predict class labels for samples in dense Matrix X. |
| [`ComplementNB.serialize()`](#serialize) | Serializes ComplementNB parameters and fitted state into BufferWriter. |
| [`ComplementNB.deserialize()`](#deserialize) | Deserializes ComplementNB from BufferReader. |

---

## Method Details

### `ComplementNB.copy()`

```mojo
def copy(self) -> Self
```
Returns a deep copy of the ComplementNB instance.

**Returns**: `Self`

---

### `ComplementNB.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: CSRMatrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit Complement Naive Bayes using dense feature matrix X and target labels y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `ComplementNB.predict_log_proba()`

```mojo
def predict_log_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_log_proba[feat_dtype: DType](self, X: CSRMatrix[feat_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class log probabilities for dense matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[feat_dtype]`

---

### `ComplementNB.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType](self, X: CSRMatrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[ feat_dtype ]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class probabilities for dense matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]`

---

### `ComplementNB.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType](self, X: CSRMatrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class labels for samples in dense Matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]`

---

### `ComplementNB.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes ComplementNB parameters and fitted state into BufferWriter.

---

### `ComplementNB.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes ComplementNB from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.naive_bayes import ComplementNB
from strata.core import Matrix

var cnb = ComplementNB[DType.float64](alpha=1.0)
cnb.fit(X_train, y_train)
var preds = cnb.predict(X_test)
var probs = cnb.predict_proba(X_test)
```
