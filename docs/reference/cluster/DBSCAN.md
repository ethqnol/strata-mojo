# `DBSCAN`

**Module**: [`strata.cluster`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable`  
**Source**: [`strata/cluster/dbscan.mojo`](file:////home/ewu/Code/Strata/strata/cluster/dbscan.mojo)

```mojo
struct DBSCAN[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable)
```

```mojo
from strata.cluster import DBSCAN
```

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN).**

Finds core samples in regions of high density and expands clusters from them.
Suitable for spatial clustering with arbitrary non-convex geometries and
identifies outliers as noise (label -1).

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision for spatial coordinate computation. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, eps: Float64 = 0.5, min_samples: Int = 5, metric: String = "euclidean", algorithm: String = "auto", leaf_size: Int = 30, p: Float64 = 2.0)
```

Initialize DBSCAN with hyperparameters.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`eps`** | `Float64` | — |
| **`min_samples`** | `Int` | — |
| **`metric`** | `String` | — |
| **`algorithm`** | `String` | — |
| **`leaf_size`** | `Int` | — |
| **`p`** | `Float64` | — |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`core_sample_indices_`** | Indices of core samples. |
| **`components_`** | Coordinates of each core sample of shape (n_core_samples, n_features). |
| **`labels_`** | Cluster labels for each point in dataset given to fit(). Noisy samples are -1. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`n_clusters_`** | Number of clusters found (excluding noise). |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`DBSCAN.fit()`](#fit) | Perform DBSCAN clustering from feature matrix X. |
| [`DBSCAN.fit_predict()`](#fit_predict) | Compute clusters from X and return cluster labels. |
| [`DBSCAN.predict()`](#predict) | Predict cluster labels for new points based on nearest core sample. |

---

## Method Details

### `DBSCAN.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Perform DBSCAN clustering from feature matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

---

### `DBSCAN.fit_predict()`

```mojo
def fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]
```
Compute clusters from X and return cluster labels.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `List[Int]` — List[Int]: Cluster labels for each point in X. Noise is -1.

---

### `DBSCAN.predict()`

```mojo
def predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Predict cluster labels for new points based on nearest core sample.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

**Returns**: `List[Int]` — List[Int]: Predicted cluster labels (noise is -1).
---

## Example

```mojo
from strata.cluster import DBSCAN
from strata.core import Matrix

var X = Matrix[DType.float64](6, 2)
X[0, 0] = 1.0; X[0, 1] = 2.0
X[1, 0] = 2.0; X[1, 1] = 2.0
X[2, 0] = 2.0; X[2, 1] = 3.0
X[3, 0] = 8.0; X[3, 1] = 7.0
X[4, 0] = 8.0; X[4, 1] = 8.0
X[5, 0] = 25.0; X[5, 1] = 80.0

var db = DBSCAN(eps=3.0, min_samples=2)
db.fit(X)
print("Labels:", db.labels_[0], db.labels_[3], db.labels_[5])
# Cluster 0, Cluster 1, Noise (-1)
```
