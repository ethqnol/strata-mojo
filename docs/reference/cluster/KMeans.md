# `KMeans`

**Module**: [`strata.cluster`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable`
**Source**: [`strata/cluster/kmeans.mojo`](file:////home/ewu/Code/Strata/strata/cluster/kmeans.mojo)

```mojo
struct KMeans[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable)
```

```mojo
from strata.cluster import KMeans
```

**K-Means clustering using Lloyd's or Elkan's algorithm.**

Clusters $N$ observations into $K$ disjoint geometric partitions by
minimizing within-cluster inertia (sum-of-squared Euclidean distances):
$$
\arg\min_{C} \sum_{i=1}^{N} \min_{\mu_j \in C} \|x_i - \mu_j\|_2^2
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, n_clusters: Int = 8, init: String = "k-means++", n_init: Int = 10, max_iter: Int = 300, tol: Float64 = 1e-4, algorithm: String = "lloyd", random_state: Int = 42)
```

Initialize the KMeans estimator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_clusters`** | `Int` | Number of clusters to form. Default 8. |
| **`init`** | `String` | Centroid initialization strategy ('k-means++', 'random'). Default 'k-means++'. |
| **`n_init`** | `Int` | Number of initializations to run. Default 10. |
| **`max_iter`** | `Int` | Maximum iterations per run. Default 300. |
| **`tol`** | `Float64` | Convergence tolerance threshold. Default 1e-4. |
| **`algorithm`** | `String` | Algorithm variant ('lloyd'). Default 'lloyd'. |
| **`random_state`** | `Int` | PRNG seed for deterministic centroid placement. Default 42. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`cluster_centers_`** | Coordinates of cluster centers matrix of shape $(K, D)$. |
| **`labels_`** | Labels of each point vector of length $N$. |
| **`inertia_`** | Sum of squared distances of samples to their closest cluster center. |
| **`n_iter_`** | Number of iterations run. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`KMeans.fit()`](#fit) | — |
| [`KMeans.predict()`](#predict) | — |
| [`KMeans.fit_predict()`](#fit_predict) | — |
| [`KMeans.transform()`](#transform) | — |
| [`KMeans.fit_transform()`](#fit_transform) | — |
| [`KMeans.score()`](#score) | — |

---

## Method Details

### `KMeans.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Fits cluster centroids on Dataset feature records.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

---

### `KMeans.predict()`

```mojo
def predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Predicts closest cluster assignments for a Dataset container.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

**Returns**: `List[Int]`

---

### `KMeans.fit_predict()`

```mojo
def fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `List[Int]`

---

### `KMeans.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `KMeans.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `KMeans.score()`

```mojo
def score[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Scalar[Self.compute_dtype]`
---

## Example

```mojo
from strata.cluster import KMeans
from strata.core import Matrix

var kmeans = KMeans[DType.float64](n_clusters=3, init="k-means++")
kmeans.fit(X_data)
var labels = kmeans.predict(X_data)
var distances = kmeans.transform(X_data)
```
