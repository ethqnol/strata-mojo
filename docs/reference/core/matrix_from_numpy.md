# `matrix_from_numpy`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)

```mojo
def matrix_from_numpy[dtype: DType = DType.float64](np_arr: PythonObject) -> Matrix[dtype]
```

```mojo
from strata.core import matrix_from_numpy
```

**Converts a 2D NumPy ndarray to a Strata Matrix[dtype].**
