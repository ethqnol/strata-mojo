# `matrix_to_numpy`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)

```mojo
def matrix_to_numpy[dtype: DType](matrix: Matrix[dtype]) -> PythonObject
```

```mojo
from strata.core import matrix_to_numpy
```

**Converts a Strata Matrix[dtype] to a NumPy 2D array.**
