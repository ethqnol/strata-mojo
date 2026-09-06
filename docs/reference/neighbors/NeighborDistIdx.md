# `NeighborDistIdx`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Comparable, Copyable, ImplicitlyCopyable, Movable`  
**Source**: [`strata/neighbors/base.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/base.mojo)

```mojo
struct NeighborDistIdx(Comparable, Copyable, ImplicitlyCopyable, Movable)
```

```mojo
from strata.neighbors import NeighborDistIdx
```

**Container holding a sample distance and its training dataset row index.**

---

## Constructors

```mojo
def __init__(out self, dist: Float64, idx: Int)
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`dist`** | `Float64` | — |
| **`idx`** | `Int` | — |
