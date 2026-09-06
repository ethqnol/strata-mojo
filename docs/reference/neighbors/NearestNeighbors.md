# `NearestNeighbors`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/neighbors/base.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/base.mojo)

```mojo
struct NearestNeighbors[compute_dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.neighbors import NearestNeighbors
```

**Unsupervised learner for implementing neighbor searches.**

Finds the $k$-nearest neighbors or all neighbors within a given radius
using brute-force or index-backed spatial distance metrics.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision used for distance computations. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, n_neighbors: Int = 5, radius: Float64 = 1.0, algorithm: String = "auto", metric: String = "euclidean", p: Float64 = 2.0)
```

Initialize NearestNeighbors estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_neighbors`** | `Int` | Number of nearest neighbors to query (>= 1). Default 5. |
| **`radius`** | `Float64` | Spatial neighborhood radius (> 0.0). Default 1.0. |
| **`algorithm`** | `String` | Neighbor search algorithm ('auto', 'brute'). Default 'auto'. |
| **`metric`** | `String` | Distance metric name. Default 'euclidean'. |
| **`p`** | `Float64` | Minkowski metric exponent (>= 1.0). Default 2.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`n_samples_fit_`** | Number of samples in the fitted data. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`NearestNeighbors.fit()`](#fit) | Fit the nearest neighbors estimator from the training dataset. |
| [`NearestNeighbors.kneighbors()`](#kneighbors) | Find the K-neighbors of points in X. |
| [`NearestNeighbors.radius_neighbors()`](#radius_neighbors) | Find the neighbors within a given radius of points in X. |

---

## Method Details

### `NearestNeighbors.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
```
Fit the nearest neighbors estimator from the training dataset.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

---

### `NearestNeighbors.kneighbors()`

```mojo
def kneighbors[in_dtype: DType](self, X: Matrix[in_dtype], n_neighbors: Int = -1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]
def kneighbors[in_dtype: DType = DType.float64](self, n_neighbors: Int = -1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Find the K-neighbors of points in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`n_neighbors`** | `Int` | — |

**Returns**: `Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]` — Tuple of: - Matrix[in_dtype]: Distances to neighbors with shape (n_queries, n_neighbors). - Matrix[DType.int32]: Indices of neighbors in the training dataset with shape (n_queries, n_neighbors).

---

### `NearestNeighbors.radius_neighbors()`

```mojo
def radius_neighbors[in_dtype: DType](self, X: Matrix[in_dtype], radius: Float64 = -1.0) -> Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]
```
Find the neighbors within a given radius of points in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`radius`** | `Float64` | — |

**Returns**: `Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]` — Tuple of: - List[List[Scalar[in_dtype]]]: Distances to each neighbor within radius. - List[List[Int]]: Indices of neighbors in the training dataset.
---

## Example

```mojo
from strata.neighbors import NearestNeighbors
from strata.core import Matrix

var nn = NearestNeighbors[DType.float64](n_neighbors=2)
nn.fit(X_train)
var res = nn.kneighbors(X_test)
var distances = res[0]
var indices = res[1]
```
