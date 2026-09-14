# `log_sum_exp`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/utils/math.mojo`](file:////home/ewu/Code/Strata/strata/utils/math.mojo)

```mojo
def log_sum_exp[dtype: DType = DType.float64](x: List[Scalar[dtype]]) -> Scalar[dtype]
```

```mojo
from strata.utils import log_sum_exp
```

**Compute numerically stable log-sum-exp: $\text{LSE}(x) = \ln \sum_i e^{x_i}$.**

**Returns**: `Scalar[dtype]` — Scalar[dtype]: Evaluated log-sum-exp scalar value.
