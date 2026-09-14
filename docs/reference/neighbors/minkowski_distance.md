# `minkowski_distance`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)

```mojo
def minkowski_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int, p: Float64 = 2.0) -> Float64
```

```mojo
from strata.neighbors import minkowski_distance
```

**Compute the Minkowski ($L_p$) distance between row X[row_x] and row Y[row_y].**

$$
d(u, v) = \left( \sum_{j=1}^D |u_j - v_j|^p \right)^{1/p}
$$

**Returns**: `Float64` — Float64: Minkowski distance.
