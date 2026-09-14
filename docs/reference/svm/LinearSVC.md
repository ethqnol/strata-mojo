# `LinearSVC`

**Module**: [`strata.svm`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`  
**Source**: [`strata/svm/linear_svc.mojo`](file:////home/ewu/Code/Strata/strata/svm/linear_svc.mojo)

```mojo
struct LinearSVC[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.svm import LinearSVC
```

**Linear Support Vector Classification.**

Fits a maximum-margin linear boundary using L2 regularization and
dual coordinate descent (LIBLINEAR algorithm). Supports both hinge
loss (L1-SVM) and squared hinge loss (L2-SVM), with One-vs-Rest (OvR)
multiclass scaling.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision for floating-point matrix operations. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, C: Float64 = 1.0, loss: String = "squared_hinge", penalty: String = "l2", dual: Bool = True, tol: Float64 = 1e-4, fit_intercept: Bool = True, intercept_scaling: Float64 = 1.0, max_iter: Int = 1000, random_state: Int = 42)
```

Initialize LinearSVC with hyperparameters.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`C`** | `Float64` | — |
| **`loss`** | `String` | — |
| **`penalty`** | `String` | — |
| **`dual`** | `Bool` | — |
| **`tol`** | `Float64` | — |
| **`fit_intercept`** | `Bool` | — |
| **`intercept_scaling`** | `Float64` | — |
| **`max_iter`** | `Int` | — |
| **`random_state`** | `Int` | — |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique sorted class labels seen during fit. |
| **`coef_`** | Weights assigned to the features. Shape (1, n_features) for binary, (n_classes, n_features) for multiclass. |
| **`intercept_`** | Constants in decision function. Length 1 for binary, length n_classes for multiclass. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`n_iter_`** | Number of iterations run until convergence. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`LinearSVC.fit()`](#fit) | Fit the linear support vector classification model according to the given training data. |
| [`LinearSVC.decision_function()`](#decision_function) | Predict confidence scores for samples in X. |
| [`LinearSVC.predict()`](#predict) | Predict class labels for samples in X. |
| [`LinearSVC.predict_proba()`](#predict_proba) | Probability estimates using calibrated decision scores. |
| [`LinearSVC.serialize()`](#serialize) | Serializes LinearSVC parameters and fitted state into BufferWriter. |
| [`LinearSVC.deserialize()`](#deserialize) | Deserializes LinearSVC from BufferReader. |

---

## Method Details

### `LinearSVC.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fit the linear support vector classification model according to the given training data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `LinearSVC.decision_function()`

```mojo
def decision_function[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[Self.compute_dtype]
```
Predict confidence scores for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[Self.compute_dtype]` — Matrix[compute_dtype]: Shape (n_samples, 1) for binary classification, or (n_samples, n_classes) for multiclass classification.

---

### `LinearSVC.predict()`

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

**Returns**: `List[Int]` — List[Int]: Predicted class labels for each sample.

---

### `LinearSVC.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Probability estimates using calibrated decision scores.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Probability matrix of shape (n_samples, n_classes).

---

### `LinearSVC.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes LinearSVC parameters and fitted state into BufferWriter.

---

### `LinearSVC.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes LinearSVC from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.svm import LinearSVC
from strata.core import Matrix

var X = Matrix[DType.float64](4, 2)
X[0, 0] = 0.0; X[0, 1] = 0.0
X[1, 0] = 0.1; X[1, 1] = 0.1
X[2, 0] = 5.0; X[2, 1] = 5.0
X[3, 0] = 5.1; X[3, 1] = 5.1

var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]
var clf = LinearSVC(C=1.0)
clf.fit(X, y)
var preds = clf.predict(X)
```
