# `pairwise_distances`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)

```mojo
def pairwise_distances[dtype: DType](X: Matrix[dtype], metric: String = "euclidean", p: Float64 = 2.0) -> Matrix[dtype]
```

```mojo
from strata.neighbors import pairwise_distances
```

**Compute the self-pairwise distance matrix between all pairs of rows in X.**

Exploits symmetry $D_{i, j} = D_{j, i}$ and $D_{i, i} = 0$ for symmetric distance metrics.

**Returns**: `Matrix[dtype]` — Matrix[dtype]: Symmetric distance matrix of shape $(N, N)$.
