# `cosine_distance`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)

```mojo
def cosine_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64
```

```mojo
from strata.neighbors import cosine_distance
```

**Compute the Cosine distance between row X[row_x] and row Y[row_y].**

$$
d(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}
$$

**Returns**: `Float64` — Float64: Cosine distance in range $[0, 2]$.
