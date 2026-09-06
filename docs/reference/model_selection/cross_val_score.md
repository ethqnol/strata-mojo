# `cross_val_score`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/model_selection/validation.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/validation.mojo)

```mojo
def cross_val_score[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], splits: List[Split], scoring: String = "accuracy") -> List[Float64]
```

```mojo
from strata.model_selection import cross_val_score
```

**Evaluates classification scores by cross-validation on pre-defined splits.**
