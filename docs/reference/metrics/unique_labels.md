# `unique_labels`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)

```mojo
def unique_labels[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> List[Float64]
```

```mojo
from strata.metrics import unique_labels
```

**Sorted list of the distinct labels appearing in y_true or y_pred.**
