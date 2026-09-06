# `GaussianNB`

**Module**: [`strata.naive_bayes`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`  
**Source**: [`strata/naive_bayes/gaussian.mojo`](file:////home/ewu/Code/Strata/strata/naive_bayes/gaussian.mojo)

```mojo
struct GaussianNB[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.naive_bayes import GaussianNB
```

**Gaussian Naive Bayes classifier for continuous feature modeling.**

Assumes that the likelihood of continuous features within each class is
normally distributed:
$$
P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{c, j}^2}} \exp\left(-\frac{(x_j - \mu_{c, j})^2}{2\sigma_{c, j}^2}\right)
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, var_smoothing: Scalar[Self.compute_dtype] = 1e-9, priors: List[Scalar[Self.compute_dtype]] = List[ Scalar[Self.compute_dtype] ]())
```

Initialize the GaussianNB classifier.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`var_smoothing`** | `Scalar[Self.compute_dtype]` | Portion of maximum variance added for numerical stability. Must be non-negative. Default 1e-9. |
| **`priors`** | `List[Scalar[Self.compute_dtype]]` | Prior probabilities of the classes. Default empty. |

```mojo
def __init__(out self, *, deinit move: Self)
```

Moves an existing GaussianNB instance.

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique class labels observed during fit. |
| **`class_count_`** | Number of training samples observed in each class. |
| **`class_prior_`** | Probability of each class. |
| **`theta_`** | Mean of each feature per class of shape $(K, D)$. |
| **`var_`** | Variance of each feature per class with variance smoothing added, of shape $(K, D)$. |
| **`epsilon_`** | Absolute additive variance smoothing factor. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`GaussianNB.fit()`](#fit) | Fit Gaussian Naive Bayes according to X, y. |
| [`GaussianNB.predict_log_proba()`](#predict_log_proba) | Predict class log probabilities for samples in X. |
| [`GaussianNB.predict_proba()`](#predict_proba) | Predict class probability distributions for samples in X. |
| [`GaussianNB.predict()`](#predict) | Predict class labels for samples in X. |
| [`GaussianNB.serialize()`](#serialize) | Serializes GaussianNB state into BufferWriter. |
| [`GaussianNB.deserialize()`](#deserialize) | Deserializes GaussianNB from BufferReader. |

---

## Method Details

### `GaussianNB.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit Gaussian Naive Bayes according to X, y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `GaussianNB.predict_log_proba()`

```mojo
def predict_log_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
```
Predict class log probabilities for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Log-probability matrix of shape $(N, K)$.

---

### `GaussianNB.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class probability distributions for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Normalized probability matrix of shape $(N, K)$, where each row sums to 1.0.

---

### `GaussianNB.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict class labels for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]` — List[Int]: Predicted class labels of length $N$.

---

### `GaussianNB.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes GaussianNB state into BufferWriter.

---

### `GaussianNB.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes GaussianNB from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.naive_bayes import GaussianNB
from strata.core import Matrix

var gnb = GaussianNB[DType.float64](var_smoothing=1e-9)
gnb.fit(X_train, y_train)
var preds = gnb.predict(X_test)
var probs = gnb.predict_proba(X_test)
```
