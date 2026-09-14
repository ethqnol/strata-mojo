# `SVDResult`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/core/linalg.mojo`](file:////home/ewu/Code/Strata/strata/core/linalg.mojo)

```mojo
struct SVDResult[dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.core import SVDResult
```

**Result of Singular Value Decomposition ($A = U \Sigma V^T$).**

---

## Constructors

```mojo
def __init__(out self, var U: Matrix[Self.dtype], var S: List[Scalar[Self.dtype]], var Vt: Matrix[Self.dtype])
```

Initialize an SVDResult container.

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`U`** | Left singular vectors matrix of shape $(M, K)$. |
| **`S`** | Singular values vector of length $K$ in descending order. |
| **`Vt`** | Right singular vectors transposed matrix of shape $(K, N)$. |
