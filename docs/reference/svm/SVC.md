# `SVC`

**Module**: [`strata.svm`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`  
**Source**: [`strata/svm/svc.mojo`](file:////home/ewu/Code/strata-mojo/strata/svm/svc.mojo)

```mojo
struct SVC[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.svm import SVC
```

**C-Support Vector Classification using non-linear kernels and SMO.**

Fits maximum-margin non-linear decision boundaries using Sequential
Minimal Optimization (SMO) across arbitrary kernel mappings (RBF,
Polynomial, Sigmoid, and Linear), with One-vs-Rest (OvR) multiclass scaling.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Floating point precision for kernel operations. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, C: Float64 = 1.0, kernel: String = "rbf", degree: Int = 3, gamma: String = "scale", gamma_val: Float64 = -1.0, coef0: Float64 = 0.0, tol: Float64 = 1e-3, max_iter: Int = 10000, decision_function_shape: String = "ovr")
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`C`** | `Float64` | — |
| **`kernel`** | `String` | — |
| **`degree`** | `Int` | — |
| **`gamma`** | `String` | — |
| **`gamma_val`** | `Float64` | — |
| **`coef0`** | `Float64` | — |
| **`tol`** | `Float64` | — |
| **`max_iter`** | `Int` | — |
| **`decision_function_shape`** | `String` | — |

```mojo
def __init__(out self, *, deinit move: Self)
```

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Unique sorted class labels seen during fit. |
| **`support_vectors_`** | Extracted support vectors of shape (n_support_vectors, n_features). |
| **`support_indices_`** | Row indices of support vectors in original training set. |
| **`dual_coef_`** | Dual coefficients alpha_i * y_i for support vectors. Shape (1, n_support_vectors) for binary, (n_classes, n_support_vectors) for multiclass. |
| **`intercept_`** | Bias terms b of length 1 for binary, length n_classes for multiclass. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`n_iter_`** | Maximum iterations reached across binary solver subproblems. |
| **`gamma_`** | Evaluated kernel scale coefficient. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`SVC.fit()`](#fit) | Fits the kernel Support Vector Classifier on feature matrix X and labels y. |
| [`SVC.decision_function()`](#decision_function) | Predict confidence scores for samples in X. |
| [`SVC.predict()`](#predict) | Predict class labels for samples in X. |
| [`SVC.predict_proba()`](#predict_proba) | Probability estimates using calibrated decision scores. |
| [`SVC.serialize()`](#serialize) | Serializes SVC parameters and fitted state into BufferWriter. |
| [`SVC.deserialize()`](#deserialize) | Deserializes SVC from BufferReader. |

---

## Method Details

### `SVC.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the kernel Support Vector Classifier on feature matrix X and labels y.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `SVC.decision_function()`

```mojo
def decision_function[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[Self.compute_dtype]
```
Predict confidence scores for samples in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |

**Returns**: `Matrix[Self.compute_dtype]` — Matrix[compute_dtype]: Shape (n_samples, 1) for binary, or (n_samples, n_classes) for multiclass.

---

### `SVC.predict()`

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

**Returns**: `List[Int]` — List[Int]: Predicted discrete class labels for each sample.

---

### `SVC.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[ feat_dtype ]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Probability estimates using calibrated decision scores.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Probability matrix of shape (n_samples, n_classes).

---

### `SVC.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes SVC parameters and fitted state into BufferWriter.

---

### `SVC.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes SVC from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.svm import SVC
from strata.core import Matrix

var clf = SVC[DType.float64](C=1.0, kernel="rbf")
clf.fit(X_train, y_train)
var preds = clf.predict(X_test)
```
