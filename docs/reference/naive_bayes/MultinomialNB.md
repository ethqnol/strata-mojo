# `MultinomialNB`

**Module**: [`strata.naive_bayes`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`
**Source**: [`strata/naive_bayes/multinomial.mojo`](file:////home/ewu/Code/Strata/strata/naive_bayes/multinomial.mojo)

```mojo
struct MultinomialNB[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.naive_bayes import MultinomialNB
```

**Multinomial Naive Bayes classifier for multinomially distributed count data.**

Suitable for discrete features (e.g. word counts for text classification):
$$
P(x \mid y = c) \propto \prod_{j=1}^{D} \theta_{c, j}^{x_j}
$$
where the smoothed feature probabilities are:
$$
\theta_{c, j} = \frac{N_{c, j} + \alpha}{N_c + \alpha \cdot D}
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, alpha: Scalar[Self.compute_dtype] = 1.0, fit_prior: Bool = True, class_prior: List[Scalar[Self.compute_dtype]] = List[ Scalar[Self.compute_dtype] ]())
```

Initialize the MultinomialNB classifier.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`alpha`** | `Scalar[Self.compute_dtype]` | Additive smoothing parameter (>= 0). Default 1.0. |
| **`fit_prior`** | `Bool` | Whether to learn class prior probabilities. Default True. |
| **`class_prior`** | `List[Scalar[Self.compute_dtype]]` | Prior probabilities of the classes. Default empty. |

```mojo
def __init__(out self, *, deinit move: Self)
```

Moves an existing MultinomialNB instance.

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique class labels observed during fit. |
| **`class_count_`** | Number of training samples observed in each class. |
| **`class_log_prior_`** | Smoothed empirical log-probability of each class. |
| **`feature_count_`** | Number of samples encountered for each (class, feature) pair of shape $(K, D)$. |
| **`feature_log_prob_`** | Empirical log probability of features given a class, $\ln P(x_j \mid y=c)$, of shape $(K, D)$. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`MultinomialNB.fit()`](#fit) | Fit Multinomial Naive Bayes using dense feature matrix X and target labels y. |
| [`MultinomialNB.predict_log_proba()`](#predict_log_proba) | Predict class log probabilities for dense matrix X. |
| [`MultinomialNB.predict_proba()`](#predict_proba) | Predict class probability distributions for dense matrix X. |
| [`MultinomialNB.predict()`](#predict) | Predict class labels for dense feature matrix X. |
| [`MultinomialNB.serialize()`](#serialize) | Serializes MultinomialNB state into BufferWriter. |
| [`MultinomialNB.deserialize()`](#deserialize) | Deserializes MultinomialNB from BufferReader. |

---

## Method Details

### `MultinomialNB.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: CSRMatrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit Multinomial Naive Bayes using dense feature matrix X and target labels y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `MultinomialNB.predict_log_proba()`

```mojo
def predict_log_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_log_proba[feat_dtype: DType](self, X: CSRMatrix[feat_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class log probabilities for dense matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Log probability matrix of shape $(N, K)$.

---

### `MultinomialNB.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType](self, X: CSRMatrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class probability distributions for dense matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Normalized probability matrix of shape $(N, K)$.

---

### `MultinomialNB.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType](self, X: CSRMatrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class labels for dense feature matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]` — List[Int]: Predicted class labels of length $N$.

---

### `MultinomialNB.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes MultinomialNB state into BufferWriter.

---

### `MultinomialNB.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes MultinomialNB from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.naive_bayes import MultinomialNB
from strata.core import Matrix

var mnb = MultinomialNB[DType.float64](alpha=1.0)
mnb.fit(X_train, y_train)
var preds = mnb.predict(X_test)
var probs = mnb.predict_proba(X_test)
```
