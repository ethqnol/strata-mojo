# `RandomForestRegressor`

**Module**: [`strata.ensemble`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor, Serializable`
**Source**: [`strata/ensemble/forest.mojo`](file:////home/ewu/Code/Strata/strata/ensemble/forest.mojo)

```mojo
struct RandomForestRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor, Serializable)
```

```mojo
from strata.ensemble import RandomForestRegressor
```

**Random Forest Regressor ensemble estimator.**

An ensemble of decision trees trained via bootstrap aggregation (bagging).
Predictions are computed as the arithmetic mean of individual tree predictions.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, n_estimators: Int = 100, criterion: String = "squared_error", max_depth: Int = -1, min_samples_split: Int = 2, min_samples_leaf: Int = 1, min_impurity_decrease: Float64 = 0.0, max_features: String = "sqrt", max_features_count: Int = -1, max_features_ratio: Float64 = 0.0, bootstrap: Bool = True, max_samples_ratio: Float64 = 1.0, max_samples_count: Int = -1, oob_score: Bool = False, random_state: Int = 42)
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
| **`n_features_in_`** | Number of features seen during fit. |
| **`feature_importances_`** | Normalized impurity feature importance vector. |
| **`oob_score_`** | Out-of-bag $R^2$ score (available when oob_score=True). |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`RandomForestRegressor.fit()`](#fit) | Fits the random forest on (X, y). |
| [`RandomForestRegressor.predict()`](#predict) | Predicts regression targets as the arithmetic mean across all tree predictions. |
| [`RandomForestRegressor.get_n_estimators()`](#get_n_estimators) | Returns the number of fitted trees. |
| [`RandomForestRegressor.get_feature_importances()`](#get_feature_importances) | Returns normalized MDI feature importances (sums to 1.0). |
| [`RandomForestRegressor.get_oob_score()`](#get_oob_score) | Returns out-of-bag R² score. Requires oob_score=True and bootstrap=True. |
| [`RandomForestRegressor.serialize()`](#serialize) | Serializes RandomForestRegressor parameters and fitted state into BufferWriter. |
| [`RandomForestRegressor.deserialize()`](#deserialize) | Deserializes RandomForestRegressor from BufferReader. |

---

## Method Details

### `RandomForestRegressor.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Fits the random forest on (X, y).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

---

### `RandomForestRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```
> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.

Predicts regression targets as the arithmetic mean across all tree predictions.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))*  |

**Returns**: `List[Scalar[feat_dtype]]`

---

### `RandomForestRegressor.get_n_estimators()`

```mojo
def get_n_estimators(self) -> Int
```
Returns the number of fitted trees.

**Returns**: `Int`

---

### `RandomForestRegressor.get_feature_importances()`

```mojo
def get_feature_importances(self) -> List[Float64]
```
Returns normalized MDI feature importances (sums to 1.0).

**Returns**: `List[Float64]`

---

### `RandomForestRegressor.get_oob_score()`

```mojo
def get_oob_score(self) -> Float64
```
Returns out-of-bag R² score. Requires oob_score=True and bootstrap=True.

**Returns**: `Float64`

---

### `RandomForestRegressor.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes RandomForestRegressor parameters and fitted state into BufferWriter.

---

### `RandomForestRegressor.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes RandomForestRegressor from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.ensemble import RandomForestRegressor
from strata.core import Matrix

var rf = RandomForestRegressor[DType.float64](n_estimators=50, max_depth=6)
rf.fit(X_train, y_train)
var preds = rf.predict(X_test)
```
