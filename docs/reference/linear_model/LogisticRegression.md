# `LogisticRegression`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`  
**Source**: [`strata/linear_model/logistic_regression.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/logistic_regression.mojo)

```mojo
struct LogisticRegression[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.linear_model import LogisticRegression
```

**Logistic Regression classifier with L2 regularization.**

Supports binary and multiclass (multinomial) classification by minimizing
the regularized cross-entropy loss with gradient optimization:
$$
\min_{W, b} -\frac{1}{N} \sum_{i=1}^{N} \ln P(y_i \mid x_i; W, b) + \frac{1}{2C} \|W\|_F^2
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, penalty: String = "l2", C: Scalar[Self.compute_dtype] = 1.0, fit_intercept: Bool = True, max_iter: Int = 100, tol: Scalar[Self.compute_dtype] = 1e-4, learning_rate: Scalar[Self.compute_dtype] = 0.1)
```

Initialize the LogisticRegression estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`penalty`** | `String` | Regularization norm ('l2' or 'none'). Default 'l2'. |
| **`C`** | `Scalar[Self.compute_dtype]` | Inverse regularization strength (must be strictly positive). Default 1.0. |
| **`fit_intercept`** | `Bool` | Whether to calculate the intercept bias term. Default True. |
| **`max_iter`** | `Int` | Maximum number of optimization iterations. Default 100. |
| **`tol`** | `Scalar[Self.compute_dtype]` | Tolerance for stopping criterion. Default 1e-4. |
| **`learning_rate`** | `Scalar[Self.compute_dtype]` | Initial step size for gradient updates. Default 0.1. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Sorted list of unique class labels seen during fit. |
| **`coef_`** | Learned weight coefficient matrix of shape $(K, D)$. |
| **`intercept_`** | Learned bias intercept vector of length $K$. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`LogisticRegression.fit()`](#fit) | Fits the logistic regression model on training data (X, y). |
| [`LogisticRegression.predict_proba()`](#predict_proba) | Predict class probability distributions for samples in X. |
| [`LogisticRegression.predict()`](#predict) | Predict discrete class labels for samples in X. |
| [`LogisticRegression.serialize()`](#serialize) | Serializes LogisticRegression parameters and fitted state into BufferWriter. |
| [`LogisticRegression.deserialize()`](#deserialize) | Deserializes LogisticRegression from BufferReader. |

---

## Method Details

### `LogisticRegression.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the logistic regression model on training data (X, y).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `LogisticRegression.predict_proba()`

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

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Probability matrix of shape $(N, K)$, where row $i$ contains the normalized probability distribution over $K$ classes.

---

### `LogisticRegression.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predict discrete class labels for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]` — List[Int]: Predicted class labels vector of length $N$.

---

### `LogisticRegression.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes LogisticRegression parameters and fitted state into BufferWriter.

---

### `LogisticRegression.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes LogisticRegression from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.linear_model import LogisticRegression
from strata.core import Matrix

var clf = LogisticRegression[DType.float64](C=1.0, max_iter=200)
clf.fit(X_train, y_train)
var probs = clf.predict_proba(X_test)
var preds = clf.predict(X_test)
```
