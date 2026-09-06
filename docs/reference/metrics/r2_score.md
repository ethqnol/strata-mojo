# `r2_score`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/metrics/regression.mojo`](file:////home/ewu/Code/Strata/strata/metrics/regression.mojo)

```mojo
def r2_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Float64
```

```mojo
from strata.metrics import r2_score
```

**Compute $R^2$ (coefficient of determination) regression score function.**

$$
R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}
$$

**Returns**: `Float64` — Float64: $R^2$ score (best possible score is 1.0, can be negative for arbitrarily worse models).
