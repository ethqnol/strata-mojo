# `check_finite`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)

```mojo
def check_finite[dtype: DType](values: List[Scalar[dtype]], name: String, caller: String)
```

```mojo
from strata.utils import check_finite
```

**Rejects NaN and infinite entries in a target or prediction list.**
