# `row_distance`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)

```mojo
def row_distance[dtype: DType](X: Matrix[dtype], row_x: Int, Y: Matrix[dtype], row_y: Int, metric: String = "euclidean", p: Float64 = 2.0) -> Scalar[dtype]
```

```mojo
from strata.neighbors import row_distance
```

**Compute pairwise distance between sample row X[row_x] and sample row Y[row_y] under metric.**

Supported metrics:
- `"euclidean"` or `"l2"`
- `"sqeuclidean"`
- `"manhattan"`, `"cityblock"`, or `"l1"`
- `"chebyshev"`, `"infinity"`, or `"max"`
- `"minkowski"` (with order parameter `p >= 1.0`)
- `"cosine"`

**Returns**: `Scalar[dtype]` — Scalar[dtype]: Calculated distance scalar.
