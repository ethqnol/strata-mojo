# `BernoulliNB`

**Module**: [`strata.naive_bayes`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`  
**Source**: [`strata/naive_bayes/bernoulli.mojo`](file:////home/ewu/Code/Strata/strata/naive_bayes/bernoulli.mojo)

```mojo
struct BernoulliNB[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.naive_bayes import BernoulliNB
```

**Bernoulli Naive Bayes classifier for multivariate Bernoulli models.**

Suitable for discrete data where features are independent binary variables
(e.g., word occurrence in text classification). Real-valued data can be
binarized using the `binarize` threshold.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, alpha: Scalar[Self.compute_dtype] = 1.0, binarize: Scalar[Self.compute_dtype] = 0.0, use_binarize: Bool = True, fit_prior: Bool = True, class_prior: List[Scalar[Self.compute_dtype]] = List[ Scalar[Self.compute_dtype] ]())
```

Initialize the BernoulliNB classifier.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`alpha`** | `Scalar[Self.compute_dtype]` | Additive smoothing parameter (>= 0). Default 1.0. |
| **`binarize`** | `Scalar[Self.compute_dtype]` | Threshold for binarizing features (features > threshold = 1). Default 0.0. |
| **`use_binarize`** | `Bool` | Whether to apply binarization threshold. Default True. |
| **`fit_prior`** | `Bool` | Whether to learn class prior probabilities. Default True. |
| **`class_prior`** | `List[Scalar[Self.compute_dtype]]` | Prior probabilities of the classes. Default empty. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique class labels observed during fit. |
| **`class_count_`** | Number of training samples observed in each class. |
| **`class_log_prior_`** | Smoothed empirical log-probability of each class. |
| **`feature_count_`** | Number of samples encountered for each (class, feature) pair where feature value was 1, of shape $(K, D)$. |
| **`feature_log_prob_`** | Empirical log probability of features given a class, $\ln P(x_j = 1 \mid y=c)$, of shape $(K, D)$. |
| **`neg_feature_log_prob_`** | Empirical log probability of absent features given a class, $\ln P(x_j = 0 \mid y=c)$, of shape $(K, D)$. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`BernoulliNB.copy()`](#copy) | Returns a deep copy of the BernoulliNB instance. |
| [`BernoulliNB.fit()`](#fit) | Fit Bernoulli Naive Bayes using dense feature matrix X and target labels y. |
| [`BernoulliNB.predict_log_proba()`](#predict_log_proba) | Predict class log probabilities for dense matrix X. |
| [`BernoulliNB.predict_proba()`](#predict_proba) | Predict class probabilities for dense matrix X. |
| [`BernoulliNB.predict()`](#predict) | Predict class labels for samples in dense Matrix X. |
| [`BernoulliNB.serialize()`](#serialize) | Serializes BernoulliNB parameters and fitted state into BufferWriter. |
| [`BernoulliNB.deserialize()`](#deserialize) | Deserializes BernoulliNB from BufferReader. |

---

## Method Details

### `BernoulliNB.copy()`

```mojo
def copy(self) -> Self
```
Returns a deep copy of the BernoulliNB instance.

**Returns**: `Self`

---

### `BernoulliNB.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: CSRMatrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit Bernoulli Naive Bayes using dense feature matrix X and target labels y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `BernoulliNB.predict_log_proba()`

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

### `BernoulliNB.predict_proba()`

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

### `BernoulliNB.predict()`

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

### `BernoulliNB.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes BernoulliNB parameters and fitted state into BufferWriter.

---

### `BernoulliNB.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes BernoulliNB from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.naive_bayes import BernoulliNB
from strata.core import Matrix

var bnb = BernoulliNB[DType.float64](alpha=1.0, binarize=0.0)
bnb.fit(X_train, y_train)
var preds = bnb.predict(X_test)
var probs = bnb.predict_proba(X_test)
```
