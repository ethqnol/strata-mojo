# `silhouette_score`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/metrics/cluster.mojo`](file:////home/ewu/Code/Strata/strata/metrics/cluster.mojo)

```mojo
def silhouette_score[dtype: DType = DType.float64](X: Matrix[dtype], labels: List[Int]) -> Float64
```

```mojo
from strata.metrics import silhouette_score
```

**Compute the mean Silhouette Coefficient of all samples.**

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance.
Samples alone in their cluster score 0.0.

**Returns**: `Float64` — Float64: Mean Silhouette Coefficient between -1.0 and 1.0.
