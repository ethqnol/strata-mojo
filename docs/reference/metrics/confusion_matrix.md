# `confusion_matrix`

**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)

```mojo
def confusion_matrix[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Matrix[DType.int64]
```

```mojo
from strata.metrics import confusion_matrix
```

**Confusion matrix C where C[i, j] counts samples of label i predicted as label j.**

Rows and columns are indexed by the sorted distinct labels.
