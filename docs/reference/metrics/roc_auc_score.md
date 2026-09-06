# `roc_auc_score`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)

```mojo
def roc_auc_score[true_dtype: DType = DType.float64, score_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_score: List[Scalar[score_dtype]], pos_label: Float64 = 1.0) -> Float64
```

```mojo
from strata.metrics import roc_auc_score
```

**Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).**

$$
\text{ROC AUC} = \frac{R_1 - \frac{n_1(n_1 + 1)}{2}}{n_1 n_0}
$$

**Returns**: `Float64` — Float64: The area under the ROC curve, between 0.0 and 1.0.
