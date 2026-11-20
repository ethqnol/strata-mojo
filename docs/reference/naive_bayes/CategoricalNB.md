# `CategoricalNB`

**Module**: [`strata.naive_bayes`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`  
**Source**: [`strata/naive_bayes/categorical.mojo`](file:////home/ewu/Code/strata-mojo/strata/naive_bayes/categorical.mojo)

```mojo
struct CategoricalNB[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.naive_bayes import CategoricalNB
```

**Categorical Naive Bayes classifier for categorically distributed features.**

Suitable for discrete classification with features representing integer-coded
categories ($0, 1, \dots, n_j - 1$). The smoothed probability of category $t$
in feature $j$ given class $c$ is:
$$
P(x_j = t \mid y = c) = \frac{N_{c, j, t} + \alpha}{N_c + \alpha \cdot n_j}
$$
where $n_j$ is the number of available categories for feature $j$.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, alpha: Scalar[Self.compute_dtype] = 1.0, fit_prior: Bool = True, class_prior: List[Scalar[Self.compute_dtype]] = List[ Scalar[Self.compute_dtype] ](), min_categories: List[Int] = List[Int]())
```

Initialize the CategoricalNB classifier.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`alpha`** | `Scalar[Self.compute_dtype]` | Additive smoothing parameter (>= 0). Default 1.0. |
| **`fit_prior`** | `Bool` | Whether to learn class prior probabilities. Default True. |
| **`class_prior`** | `List[Scalar[Self.compute_dtype]]` | Prior probabilities of the classes. Default empty. |
| **`min_categories`** | `List[Int]` | Minimum number of categories per feature. Default empty. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique sorted class labels observed during fit. |
| **`class_count_`** | Number of training samples observed in each class. |
| **`class_log_prior_`** | Smoothed empirical log-probability of each class. |
| **`category_count_`** | Per-feature matrices of shape $(K, n_j)$ holding sample counts for each (class, category) pair. |
| **`feature_log_prob_`** | Per-feature matrices of shape $(K, n_j)$ holding empirical log probability $\ln P(x_j = t \mid y = c)$. |
| **`n_categories_`** | Number of categories $n_j$ used for each feature. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`CategoricalNB.copy()`](#copy) | Return a deep copy of the CategoricalNB instance. |
| [`CategoricalNB.fit()`](#fit) | Fit Categorical Naive Bayes using feature matrix X and target labels y. |
| [`CategoricalNB.predict_log_proba()`](#predict_log_proba) | Predict class log probabilities for samples in X. |
| [`CategoricalNB.predict_proba()`](#predict_proba) | Predict class probability distributions for samples in X. |
| [`CategoricalNB.predict()`](#predict) | Predict class labels for samples in X. |
| [`CategoricalNB.serialize()`](#serialize) | Serialize CategoricalNB parameters and fitted state into BufferWriter. |
| [`CategoricalNB.deserialize()`](#deserialize) | Deserialize CategoricalNB from BufferReader. |

---

## Method Details

### `CategoricalNB.copy()`

```mojo
def copy(self) -> Self
```
Return a deep copy of the CategoricalNB instance.

**Returns**: `Self`

---

### `CategoricalNB.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit Categorical Naive Bayes using feature matrix X and target labels y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Non-negative integer-coded categorical matrix of shape $(N, D)$. |
| **`y`** | `List[Scalar[target_dtype]]` | Target class labels of length $N$. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `CategoricalNB.predict_log_proba()`

```mojo
def predict_log_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
```
Predict class log probabilities for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Categorical feature matrix of shape $(N, D)$. |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Log-probability matrix of shape $(N, K)$.

---

### `CategoricalNB.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class probability distributions for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Categorical feature matrix of shape $(N, D)$. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Normalized probability matrix of shape $(N, K)$, where each row sums to 1.0.

---

### `CategoricalNB.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class labels for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Categorical feature matrix of shape $(N, D)$. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]` — List[Int]: Predicted class labels of length $N$.

---

### `CategoricalNB.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serialize CategoricalNB parameters and fitted state into BufferWriter.

---

### `CategoricalNB.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserialize CategoricalNB from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.naive_bayes import CategoricalNB
from strata.core import Matrix

var cnb = CategoricalNB[DType.float64](alpha=1.0)
cnb.fit(X_train, y_train)
var preds = cnb.predict(X_test)
var probs = cnb.predict_proba(X_test)
```
