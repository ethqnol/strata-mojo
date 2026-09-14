# `accuracy_score`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)

```mojo
def accuracy_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]], normalize: Bool = True) -> Float64
```

```mojo
from strata.metrics import accuracy_score
```

**Fraction (or count, if normalize is False) of correctly classified samples.**
