# `GaussianSample`

**Module**: [`strata.mixture`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/mixture/gaussian_mixture.mojo`](file:////home/ewu/Code/strata-mojo/strata/mixture/gaussian_mixture.mojo)

```mojo
struct GaussianSample[dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.mixture import GaussianSample
```

**Generated synthetic samples and component labels from GaussianMixture.sample.**

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`dtype`** | Scalar floating-point precision type. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, var X: Matrix[Self.dtype], var y: List[Int])
```

Initialize a GaussianSample container.

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`X`** | Sampled observations matrix of shape $(N, D)$. |
| **`y`** | Originating mixture component index for each sample of length $N$. |
