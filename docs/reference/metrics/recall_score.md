# `recall_score`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)

```mojo
def recall_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]], average: String = "binary", pos_label: Float64 = 1.0, zero_division: Float64 = 0.0) -> Float64
```

```mojo
from strata.metrics import recall_score
```

**Compute classification recall (sensitivity) score.**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Returns**: `Float64` — Float64: Recall score ratio.
