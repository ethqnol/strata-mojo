# `f1_score`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)

```mojo
def f1_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]], average: String = "binary", pos_label: Float64 = 1.0, zero_division: Float64 = 0.0) -> Float64
```

```mojo
from strata.metrics import f1_score
```

**Compute classification F1 score (harmonic mean of precision and recall).**

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}
$$

**Returns**: `Float64` — Float64: F1 score between 0.0 and 1.0.
