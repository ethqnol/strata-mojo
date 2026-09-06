# `KDTree`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`
**Source**: [`strata/neighbors/kd_tree.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/kd_tree.mojo)

```mojo
struct KDTree[compute_dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.neighbors import KDTree
```

**Fast spatial index for nearest neighbor and radius queries in low dimensions.**

Organizes $N$ points in $D$-dimensional space into a binary space-partitioning
tree for $O(K \log N)$ neighbor lookups.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision for spatial coordinate representation. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, data: Matrix[Self.compute_dtype], metric: String = "euclidean")
```

Construct a KDTree from a matrix of spatial coordinates.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`data`** | `Matrix[Self.compute_dtype]` | Input point matrix of shape (n_samples, n_features). |
| **`metric`** | `String` | Distance metric ('euclidean', 'manhattan', 'chebyshev'). Default 'euclidean'. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`n_samples_`** | Number of samples indexed in the tree. |
| **`n_features_`** | Dimensionality of the indexed space. |
| **`root_idx_`** | Index of the root node in the internal buffer. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`KDTree.query()`](#query) | Query the KDTree for the k-nearest neighbors of points in X. |
| [`KDTree.query_radius()`](#query_radius) | Find all points within distance r of points in X. |

---

## Method Details

### `KDTree.query()`

```mojo
def query[in_dtype: DType](self, X: Matrix[in_dtype], k: Int = 1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]
```
Query the KDTree for the k-nearest neighbors of points in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`k`** | `Int` | — |

**Returns**: `Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]` — Tuple of: - Matrix[in_dtype]: Distances to neighbors with shape (n_queries, k). - Matrix[DType.int32]: Indices of neighbors in original data with shape (n_queries, k).

---

### `KDTree.query_radius()`

```mojo
def query_radius[in_dtype: DType](self, X: Matrix[in_dtype], r: Float64) -> Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]
```
Find all points within distance r of points in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`r`** | `Float64` | — |

**Returns**: `Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]` — Tuple of: - List[List[Scalar[in_dtype]]]: Distances to each neighbor within radius. - List[List[Int]]: Dataset indices of neighbors within radius.
---

## Example

```mojo
from strata.neighbors import KDTree
from strata.core import Matrix

var tree = KDTree[DType.float64](X_train, metric="euclidean")
var res = tree.query(X_query, k=3)
var dists = res[0]
var idxs = res[1]
```
