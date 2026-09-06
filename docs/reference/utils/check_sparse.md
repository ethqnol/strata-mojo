# `check_sparse`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`
**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)

```mojo
def check_sparse[dtype: DType](rows: Int, cols: Int, data: List[Scalar[dtype]], indices: List[Int], indptr: List[Int], is_csr: Bool = True, allow_empty: Bool = True, caller: String = "SparseMatrix.__init__")
```

```mojo
from strata.utils import check_sparse
```

**Validates CSR/CSC sparse matrix format invariants.**
