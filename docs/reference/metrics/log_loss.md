# `log_loss`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)

```mojo
def log_loss[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: Matrix[pred_dtype], normalize: Bool = True) -> Float64
```

```mojo
from strata.metrics import log_loss
```

**Compute log loss (cross-entropy loss), the negative log-likelihood of true labels.**

$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i, k} \log(p_{i, k})
$$

**Returns**: `Float64` — Float64: The mean (or total) cross-entropy between y_true and y_pred.
