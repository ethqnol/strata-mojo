# `PCA`

**Module**: [`strata.decomposition`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/decomposition/pca.mojo`](file:////home/ewu/Code/Strata/strata/decomposition/pca.mojo)

```mojo
struct PCA[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.decomposition import PCA
```

**Principal Component Analysis (PCA).**

Linear dimensionality reduction using Singular Value Decomposition of the
centered data matrix to project it to a lower dimensional subspace:
$$
X_{\text{projected}} = (X - \mu) V_k
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, n_components: Int = 0, whiten: Bool = False)
```

Initialize the PCA transformer.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_components`** | `Int` | Number of principal components to project onto. Default 0 (all). |
| **`whiten`** | `Bool` | Whether to scale component projections by variance. Default False. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`components_`** | Principal axes in feature space, representing directions of maximum variance matrix of shape $(K, D)$. |
| **`explained_variance_`** | Variance explained by each selected component vector of length $K$. |
| **`explained_variance_ratio_`** | Percentage of variance explained by each component. |
| **`singular_values_`** | Singular values corresponding to each of the selected components. |
| **`mean_`** | Per-feature empirical mean estimated from the training set. |
| **`n_components_`** | Estimated number of components. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`PCA.fit()`](#fit) | Fits the PCA model on matrix X. |
| [`PCA.transform()`](#transform) | Projects matrix X onto the principal components. |
| [`PCA.fit_transform()`](#fit_transform) | Fits PCA to X and returns the projected data. |
| [`PCA.inverse_transform()`](#inverse_transform) | Transforms data back to its original space. |

---

## Method Details

### `PCA.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fits the PCA model on matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

---

### `PCA.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Projects matrix X onto the principal components.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]`

---

### `PCA.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]
```
> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.

Fits PCA to X and returns the projected data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix. *(Can be provided alternatively in place of X)* |

**Returns**: `Matrix[in_dtype]`

---

### `PCA.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```
Transforms data back to its original space.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]`
---

## Example

```mojo
from strata.decomposition import PCA
from strata.core import Matrix

var pca = PCA[DType.float64](n_components=2)
pca.fit(X_train)
var X_proj = pca.transform(X_train)
```
