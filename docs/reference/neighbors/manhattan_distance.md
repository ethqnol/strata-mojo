# `manhattan_distance`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)

```mojo
def manhattan_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64
```

```mojo
from strata.neighbors import manhattan_distance
```

**Compute the Manhattan ($L_1$ / taxicab / cityblock) distance between row X[row_x] and row Y[row_y].**

$$
d(u, v) = \sum_{j=1}^D |u_j - v_j|
$$

**Returns**: `Float64` — Float64: Manhattan distance.
