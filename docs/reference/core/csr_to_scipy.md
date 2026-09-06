# `csr_to_scipy`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)

```mojo
def csr_to_scipy[dtype: DType](csr: CSRMatrix[dtype]) -> PythonObject
```

```mojo
from strata.core import csr_to_scipy
```

**Converts a Strata CSRMatrix[dtype] to a scipy.sparse.csr_matrix.**
