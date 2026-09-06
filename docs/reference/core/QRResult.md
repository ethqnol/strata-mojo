# `QRResult`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`
**Source**: [`strata/core/linalg.mojo`](file:////home/ewu/Code/Strata/strata/core/linalg.mojo)

```mojo
struct QRResult[dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.core import QRResult
```

**Result of QR Decomposition ($A = Q R$).**

---

## Constructors

```mojo
def __init__(out self, var Q: Matrix[Self.dtype], var R: Matrix[Self.dtype])
```

Initialize a QRResult container.

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`Q`** | Orthogonal matrix of shape $(M, K)$. |
| **`R`** | Upper triangular matrix of shape $(K, N)$. |
