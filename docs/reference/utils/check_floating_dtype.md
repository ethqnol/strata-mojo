# `check_floating_dtype`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)

```mojo
def check_floating_dtype[dtype: DType, caller: StringLiteral = "Estimator"]()
```

```mojo
from strata.utils import check_floating_dtype
```

**Asserts at compile time that the specified dtype is a floating-point type.**
