# `RandomForestClassifier`

**Module**: [`strata.ensemble`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable, Serializable`
**Source**: [`strata/ensemble/forest.mojo`](file:////home/ewu/Code/Strata/strata/ensemble/forest.mojo)

```mojo
struct RandomForestClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable, Serializable)
```

```mojo
from strata.ensemble import RandomForestClassifier
```

**Random Forest Classifier ensemble estimator.**

An ensemble of decision trees trained via bootstrap aggregation (bagging).
Predictions are computed via soft voting (averaging predicted class probabilities
across all trees and selecting the argmax class).

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, n_estimators: Int = 100, criterion: String = "gini", max_depth: Int = -1, min_samples_split: Int = 2, min_samples_leaf: Int = 1, min_impurity_decrease: Float64 = 0.0, max_features: String = "sqrt", max_features_count: Int = -1, max_features_ratio: Float64 = 0.0, bootstrap: Bool = True, max_samples_ratio: Float64 = 1.0, max_samples_count: Int = -1, oob_score: Bool = False, random_state: Int = 42)
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_estimators`** | `Int` | — |
| **`criterion`** | `String` | — |
| **`max_depth`** | `Int` | — |
| **`min_samples_split`** | `Int` | — |
| **`min_samples_leaf`** | `Int` | — |
| **`min_impurity_decrease`** | `Float64` | — |
| **`max_features`** | `String` | — |
| **`max_features_count`** | `Int` | — |
| **`max_features_ratio`** | `Float64` | — |
| **`bootstrap`** | `Bool` | — |
| **`max_samples_ratio`** | `Float64` | — |
| **`max_samples_count`** | `Int` | — |
| **`oob_score`** | `Bool` | — |
| **`random_state`** | `Int` | — |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Sorted list of unique class labels seen during fit. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`feature_importances_`** | Normalized impurity feature importance vector. |
| **`oob_score_`** | Out-of-bag accuracy score (available when oob_score=True). |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`RandomForestClassifier.fit()`](#fit) | Fits the random forest classifier on (X, y). |
| [`RandomForestClassifier.predict_proba()`](#predict_proba) | Generates class probability estimates for input matrix X by averaging tree probabilities. |
| [`RandomForestClassifier.predict()`](#predict) | Generates discrete class predictions via soft-voting argmax over predicted class probabilities. |
| [`RandomForestClassifier.get_n_estimators()`](#get_n_estimators) | Returns the number of fitted trees. |
| [`RandomForestClassifier.get_feature_importances()`](#get_feature_importances) | Returns normalized MDI feature importances (sums to 1.0). |
| [`RandomForestClassifier.get_oob_score()`](#get_oob_score) | Returns out-of-bag accuracy score. Requires oob_score=True and bootstrap=True. |
| [`RandomForestClassifier.get_classes()`](#get_classes) | Returns the sorted list of known class labels. |
| [`RandomForestClassifier.serialize()`](#serialize) | Serializes RandomForestClassifier parameters and fitted state into BufferWriter. |
| [`RandomForestClassifier.deserialize()`](#deserialize) | Deserializes RandomForestClassifier from BufferReader. |

---

## Method Details

### `RandomForestClassifier.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the random forest classifier on (X, y).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `RandomForestClassifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Generates class probability estimates for input matrix X by averaging tree probabilities.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `Matrix[feat_dtype]`

---

### `RandomForestClassifier.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Generates discrete class predictions via soft-voting argmax over predicted class probabilities.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Int]`

---

### `RandomForestClassifier.get_n_estimators()`

```mojo
def get_n_estimators(self) -> Int
```
Returns the number of fitted trees.

**Returns**: `Int`

---

### `RandomForestClassifier.get_feature_importances()`

```mojo
def get_feature_importances(self) -> List[Float64]
```
Returns normalized MDI feature importances (sums to 1.0).

**Returns**: `List[Float64]`

---

### `RandomForestClassifier.get_oob_score()`

```mojo
def get_oob_score(self) -> Float64
```
Returns out-of-bag accuracy score. Requires oob_score=True and bootstrap=True.

**Returns**: `Float64`

---

### `RandomForestClassifier.get_classes()`

```mojo
def get_classes(self) -> List[Int]
```
Returns the sorted list of known class labels.

**Returns**: `List[Int]`

---

### `RandomForestClassifier.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes RandomForestClassifier parameters and fitted state into BufferWriter.

---

### `RandomForestClassifier.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes RandomForestClassifier from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.ensemble import RandomForestClassifier
from strata.core import Matrix

var rf = RandomForestClassifier[DType.float64](n_estimators=50, max_depth=6)
rf.fit(X_train, y_train)
var preds = rf.predict(X_test)
```
