# `csr_from_scipy`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)

```mojo
def csr_from_scipy[dtype: DType = DType.float64](sp_arr: PythonObject) -> CSRMatrix[dtype]
```

```mojo
from strata.core import csr_from_scipy
```

**Converts a scipy.sparse.csr_matrix to a Strata CSRMatrix[dtype].**
