# `SGDClassifier`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`
**Source**: [`strata/linear_model/sgd_classifier.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/sgd_classifier.mojo)

```mojo
struct SGDClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)
```

```mojo
from strata.linear_model import SGDClassifier
```

**Linear classifier with SGD training.**

Supports linear SVM (`loss='hinge'`), Logistic Regression (`loss='log_loss'`),
Modified Huber (`loss='modified_huber'`), and Squared Hinge (`loss='squared_hinge'`).
Minimizes the regularized loss:
$$
\min_{W, b} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(W x_i + b, y_i) + \alpha \mathcal{R}(W)
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, loss: String = "hinge", penalty: String = "l2", alpha: Scalar[Self.compute_dtype] = 1e-4, l1_ratio: Scalar[Self.compute_dtype] = 0.15, fit_intercept: Bool = True, max_iter: Int = 1000, tol: Scalar[Self.compute_dtype] = 1e-3, shuffle_data: Bool = True, epsilon: Scalar[Self.compute_dtype] = 0.1, random_state: Int = 42, learning_rate: String = "optimal", eta0: Scalar[Self.compute_dtype] = 0.01, power_t: Scalar[Self.compute_dtype] = 0.5)
```

Initialize the SGDClassifier estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`loss`** | `String` | — |
| **`penalty`** | `String` | — |
| **`alpha`** | `Scalar[Self.compute_dtype]` | — |
| **`l1_ratio`** | `Scalar[Self.compute_dtype]` | — |
| **`fit_intercept`** | `Bool` | — |
| **`max_iter`** | `Int` | — |
| **`tol`** | `Scalar[Self.compute_dtype]` | — |
| **`shuffle_data`** | `Bool` | — |
| **`epsilon`** | `Scalar[Self.compute_dtype]` | — |
| **`random_state`** | `Int` | — |
| **`learning_rate`** | `String` | — |
| **`eta0`** | `Scalar[Self.compute_dtype]` | — |
| **`power_t`** | `Scalar[Self.compute_dtype]` | — |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique sorted class labels observed in training data. |
| **`coef_`** | Learned weights matrix of shape $(K, D)$ (or $(1, D)$ for binary). |
| **`intercept_`** | Learned bias vector of length $K$ (or 1 for binary). |
| **`n_iter_`** | Actual number of epochs executed before convergence. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |
| **`n_features_in_`** | Number of features seen during fitting. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`SGDClassifier.fit()`](#fit) | Fit linear model classifier with Stochastic Gradient Descent. |
| [`SGDClassifier.decision_function()`](#decision_function) | Predict linear margin decision function. |
| [`SGDClassifier.predict_proba()`](#predict_proba) | Probability estimates for each class. |
| [`SGDClassifier.predict()`](#predict) | Predict class labels for samples in X. |

---

## Method Details

### `SGDClassifier.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit linear model classifier with Stochastic Gradient Descent.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `SGDClassifier.decision_function()`

```mojo
def decision_function[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
```
Predict linear margin decision function.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[feat_dtype]`

---

### `SGDClassifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Probability estimates for each class.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]`

---

### `SGDClassifier.predict()`

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

**Returns**: `List[Int]`
---

## Example

```mojo
from strata.linear_model import SGDClassifier
from strata.core import Matrix

var clf = SGDClassifier[DType.float64](loss="log_loss", penalty="l2")
clf.fit(X_train, y_train)
var preds = clf.predict(X_test)
```
