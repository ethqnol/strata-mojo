# `KDNode`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/neighbors/kd_tree.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/kd_tree.mojo)

```mojo
struct KDNode(Copyable, Movable)
```

```mojo
from strata.neighbors import KDNode
```

**Contiguous node in a flat KD-Tree buffer.**

---

## Constructors

```mojo
def __init__(out self, point_idx: Int, split_dim: Int, split_val: Float64, left_child: Int = -1, right_child: Int = -1)
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`point_idx`** | `Int` | — |
| **`split_dim`** | `Int` | — |
| **`split_val`** | `Float64` | — |
| **`left_child`** | `Int` | — |
| **`right_child`** | `Int` | — |
