# `mean_absolute_error`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/metrics/regression.mojo`](file:////home/ewu/Code/Strata/strata/metrics/regression.mojo)

```mojo
def mean_absolute_error[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Float64
```

```mojo
from strata.metrics import mean_absolute_error
```

**Compute Mean Absolute Error (MAE) regression loss.**

$$
\text{MAE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

**Returns**: `Float64` — Float64: Non-negative floating point mean absolute error.
