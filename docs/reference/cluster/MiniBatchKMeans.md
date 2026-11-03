# `MiniBatchKMeans`

**Module**: [`strata.cluster`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable`  
**Source**: [`strata/cluster/minibatch_kmeans.mojo`](file:////home/ewu/Code/strata-mojo/strata/cluster/minibatch_kmeans.mojo)

```mojo
struct MiniBatchKMeans[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable)
```

```mojo
from strata.cluster import MiniBatchKMeans
```

**Mini-Batch K-Means clustering algorithm.**

Mini-Batch K-Means uses mini-batches of samples to reduce computation time while
optimizing the same objective function as full-batch K-Means:
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
def __init__(out self, n_clusters: Int = 8, init: String = "k-means++", max_iter: Int = 100, batch_size: Int = 1024, tol: Float64 = 0.0, max_no_improvement: Int = 10, n_init: Int = 3, reassignment_ratio: Float64 = 0.01, random_state: Int = 42)
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_clusters`** | `Int` | The number of clusters to form as well as the number of centroids to generate. Default 8. |
| **`init`** | `String` | Method for initialization ('k-means++', 'random'). Default 'k-means++'. |
| **`max_iter`** | `Int` | Maximum number of mini-batch iterations. Default 100. |
| **`batch_size`** | `Int` | Size of mini-batches drawn per iteration. Default 1024. |
| **`tol`** | `Float64` | Tolerance threshold for early stopping based on center shift. Default 1e-4. |
| **`max_no_improvement`** | `Int` | Early stopping iteration count without inertia improvement. Default 10. |
| **`n_init`** | `Int` | Number of random initialization attempts. Default 3. |
| **`reassignment_ratio`** | `Float64` | Fraction of max count threshold for center reassignment. Default 0.01. |
| **`random_state`** | `Int` | PRNG seed for reproducible centroid initializations. Default 42. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`cluster_centers_`** | Coordinates of cluster centers matrix of shape $(K, D)$. |
| **`labels_`** | Labels of each point vector of length $N$. |
| **`inertia_`** | Sum of squared distances of samples to their closest cluster center. |
| **`n_iter_`** | Number of iterations run during fitting. |
| **`n_steps_`** | Total mini-batch update steps performed. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`MiniBatchKMeans.partial_fit()`](#partial_fit) | — |
| [`MiniBatchKMeans.fit()`](#fit) | — |
| [`MiniBatchKMeans.predict()`](#predict) | — |
| [`MiniBatchKMeans.fit_predict()`](#fit_predict) | — |
| [`MiniBatchKMeans.transform()`](#transform) | — |
| [`MiniBatchKMeans.fit_transform()`](#fit_transform) | — |
| [`MiniBatchKMeans.score()`](#score) | — |

---

## Method Details

### `MiniBatchKMeans.partial_fit()`

```mojo
def partial_fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

---

### `MiniBatchKMeans.fit()`

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

### `MiniBatchKMeans.predict()`

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

### `MiniBatchKMeans.fit_predict()`

```mojo
def fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `List[Int]`

---

### `MiniBatchKMeans.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `MiniBatchKMeans.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`

---

### `MiniBatchKMeans.score()`

```mojo
def score[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Scalar[Self.compute_dtype]`

---
