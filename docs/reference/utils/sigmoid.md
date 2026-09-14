# `sigmoid`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/utils/math.mojo`](file:////home/ewu/Code/Strata/strata/utils/math.mojo)

```mojo
def sigmoid[dtype: DType = DType.float64](x: Scalar[dtype]) -> Scalar[dtype]
```

```mojo
from strata.utils import sigmoid
```

**Compute the logistic sigmoid function $\sigma(x)$.**

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Returns**: `Scalar[dtype]` — Scalar[dtype]: Evaluated sigmoid activation in the range $(0, 1)$.
