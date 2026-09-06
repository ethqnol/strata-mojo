# `softmax`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/utils/math.mojo`](file:////home/ewu/Code/Strata/strata/utils/math.mojo)

```mojo
def softmax[dtype: DType = DType.float64](x: List[Scalar[dtype]]) -> List[Scalar[dtype]]
```

```mojo
from strata.utils import softmax
```

**Compute numerically stable softmax probability distribution.**

$$
\text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

**Returns**: `List[Scalar[dtype]]` — List[Scalar[dtype]]: Normalized probability distribution vector summing to 1.0.
