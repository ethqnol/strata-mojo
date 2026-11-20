# `GaussianMixture`

**Module**: [`strata.mixture`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable, Serializable`  
**Source**: [`strata/mixture/gaussian_mixture.mojo`](file:////home/ewu/Code/strata-mojo/strata/mixture/gaussian_mixture.mojo)

```mojo
struct GaussianMixture[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable, Serializable)
```

```mojo
from strata.mixture import GaussianMixture
```

**Gaussian Mixture Model trained with Expectation-Maximization (EM).**

Models a distribution as a weighted mixture of $K$ multivariate Gaussian
components:
$$
p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$
supporting `'full'`, `'tied'`, `'diag'`, and `'spherical'` covariance
structures with multi-start initialization, custom parameter priors,
density scoring, AIC/BIC model selection, and generative sampling.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, n_components: Int = 1, covariance_type: String = "full", tol: Float64 = 1e-3, reg_covar: Float64 = 1e-6, max_iter: Int = 100, n_init: Int = 1, init_params: String = "kmeans", weights_init: List[Scalar[Self.compute_dtype]] = List[ Scalar[Self.compute_dtype] ](), means_init: Matrix[Self.compute_dtype] = Matrix[Self.compute_dtype]( 0, 0, 0 ), precisions_init: List[Matrix[Self.compute_dtype]] = List[ Matrix[Self.compute_dtype] ](), random_state: Int = 42, warm_start: Bool = False)
```

Initialize the GaussianMixture model.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_components`** | `Int` | Number of Gaussian components (>= 1). Default 1. |
| **`covariance_type`** | `String` | Covariance structure ('full', 'tied', 'diag', 'spherical'). Default 'full'. |
| **`tol`** | `Float64` | EM convergence tolerance (>= 0.0). Default 1e-3. |
| **`reg_covar`** | `Float64` | Diagonal covariance regularization (>= 0.0). Default 1e-6. |
| **`max_iter`** | `Int` | Maximum EM iterations per restart (>= 1). Default 100. |
| **`n_init`** | `Int` | Number of initialization restarts (>= 1). Default 1. |
| **`init_params`** | `String` | Initialization method ('kmeans', 'k-means++', 'random', 'random_from_data'). Default 'kmeans'. |
| **`weights_init`** | `List[Scalar[Self.compute_dtype]]` | Optional initial mixture weights. Default empty. |
| **`means_init`** | `Matrix[Self.compute_dtype]` | Optional initial component means. Default empty. |
| **`precisions_init`** | `List[Matrix[Self.compute_dtype]]` | Optional initial precision matrices. Default empty. |
| **`random_state`** | `Int` | PRNG seed. Default 42. |
| **`warm_start`** | `Bool` | Reuse fitted state as initialization. Default False. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`weights_`** | Fitted mixture weights of length $K$. |
| **`means_`** | Fitted component centers of shape $(K, D)$. |
| **`covariances_`** | Fitted covariances stored per `covariance_type`. |
| **`precisions_`** | Fitted precision matrices ($\Sigma_k^{-1}$). |
| **`precisions_cholesky_`** | Upper-triangular (or reciprocal std) Cholesky factors of the precision matrices. |
| **`converged_`** | True if EM reached tolerance within `max_iter`. |
| **`n_iter_`** | Number of EM iterations executed in the winning restart. |
| **`lower_bound_`** | Final mean log-likelihood lower bound achieved on training data. |
| **`n_features_in_`** | Number of features observed during fit. |
| **`is_fitted`** | Boolean flag indicating whether the model has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`GaussianMixture.copy()`](#copy) | Return a deep copy of the GaussianMixture instance. |
| [`GaussianMixture.fit()`](#fit) | Fit Gaussian Mixture Model parameters using Expectation-Maximization. |
| [`GaussianMixture.predict()`](#predict) | Predict component labels for observations in X. |
| [`GaussianMixture.fit_predict()`](#fit_predict) | Fit the GaussianMixture on X and predict component assignments. |
| [`GaussianMixture.predict_proba()`](#predict_proba) | Predict posterior responsibility probabilities for each component. |
| [`GaussianMixture.score_samples()`](#score_samples) | Compute per-sample marginal log-likelihood $\ln p(x_i)$. |
| [`GaussianMixture.score()`](#score) | Compute the average per-sample log-likelihood over X. |
| [`GaussianMixture.aic()`](#aic) | Compute the Akaike Information Criterion (AIC) on observations X. |
| [`GaussianMixture.bic()`](#bic) | Compute the Bayesian Information Criterion (BIC) on observations X. |
| [`GaussianMixture.sample()`](#sample) | Generate random samples from the fitted Gaussian mixture distribution. |
| [`GaussianMixture.serialize()`](#serialize) | Serialize GaussianMixture parameters and fitted state into BufferWriter. |
| [`GaussianMixture.deserialize()`](#deserialize) | Deserialize GaussianMixture from BufferReader. |

---

## Method Details

### `GaussianMixture.copy()`

```mojo
def copy(self) -> Self
```
Return a deep copy of the GaussianMixture instance.

**Returns**: `Self`

---

### `GaussianMixture.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Fit Gaussian Mixture Model parameters using Expectation-Maximization.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Training feature matrix of shape $(N, D)$ with $N \ge K$. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

---

### `GaussianMixture.predict()`

```mojo
def predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Predict component labels for observations in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Input feature matrix of shape $(N, D)$. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding data records. *(Overloaded alternative)* |

**Returns**: `List[Int]` — List[Int]: Most probable component index per sample of length $N$.

---

### `GaussianMixture.fit_predict()`

```mojo
def fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]
```
Fit the GaussianMixture on X and predict component assignments.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Input feature matrix of shape $(N, D)$. |

**Returns**: `List[Int]` — List[Int]: Predicted component indices of length $N$.

---

### `GaussianMixture.predict_proba()`

```mojo
def predict_proba[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[Self.compute_dtype]
```
Predict posterior responsibility probabilities for each component.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Input feature matrix of shape $(N, D)$. |

**Returns**: `Matrix[Self.compute_dtype]` — Matrix[Self.compute_dtype]: Posterior probability matrix of shape $(N, K)$ with each row summing to 1.0.

---

### `GaussianMixture.score_samples()`

```mojo
def score_samples[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Scalar[Self.compute_dtype]]
```
Compute per-sample marginal log-likelihood $\ln p(x_i)$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Input feature matrix of shape $(N, D)$. |

**Returns**: `List[Scalar[Self.compute_dtype]]` — List[Scalar[Self.compute_dtype]]: Marginal log-densities of length $N$.

---

### `GaussianMixture.score()`

```mojo
def score[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]
```
Compute the average per-sample log-likelihood over X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Input feature matrix of shape $(N, D)$. |

**Returns**: `Scalar[Self.compute_dtype]` — Scalar[Self.compute_dtype]: Mean marginal log-likelihood across samples.

---

### `GaussianMixture.aic()`

```mojo
def aic[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]
```
Compute the Akaike Information Criterion (AIC) on observations X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Input feature matrix of shape $(N, D)$. |

**Returns**: `Scalar[Self.compute_dtype]` — Scalar[Self.compute_dtype]: Evaluated AIC value.

---

### `GaussianMixture.bic()`

```mojo
def bic[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]
```
Compute the Bayesian Information Criterion (BIC) on observations X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Input feature matrix of shape $(N, D)$. |

**Returns**: `Scalar[Self.compute_dtype]` — Scalar[Self.compute_dtype]: Evaluated BIC value.

---

### `GaussianMixture.sample()`

```mojo
def sample(self, n_samples: Int = 1, random_state: Int = -1) -> GaussianSample[Self.compute_dtype]
```
Generate random samples from the fitted Gaussian mixture distribution.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`n_samples`** | `Int` | Number of samples to draw ($\ge 1$). Default 1. |
| **`random_state`** | `Int` | Optional PRNG seed override. When < 0, uses model's `random_state`. Default -1. |

**Returns**: `GaussianSample[Self.compute_dtype]` — GaussianSample[Self.compute_dtype]: Container holding sampled matrix `X` of shape $(n\_samples, D)$ and component indices `y`.

---

### `GaussianMixture.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serialize GaussianMixture parameters and fitted state into BufferWriter.

---

### `GaussianMixture.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserialize GaussianMixture from BufferReader.

**Returns**: `Self`
---

## Example

```mojo
from strata.mixture import GaussianMixture
from strata.core import Matrix

var gmm = GaussianMixture[DType.float64](
    n_components=3, covariance_type="full", random_state=42
)
gmm.fit(X_train)
var labels = gmm.predict(X_test)
var probs = gmm.predict_proba(X_test)
```
