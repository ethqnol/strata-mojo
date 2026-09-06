# `DatasetSplit`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`
**Source**: [`strata/core/dataset.mojo`](file:////home/ewu/Code/Strata/strata/core/dataset.mojo)

```mojo
struct DatasetSplit[feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Movable)
```

```mojo
from strata.core import DatasetSplit
```

**Container holding train and test partitions of a Dataset.**

---

## Constructors

```mojo
def __init__(out self, var train: Dataset[Self.feat_dtype, Self.target_dtype], var test: Dataset[Self.feat_dtype, Self.target_dtype])
```

Initialize a DatasetSplit container.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`train`** | — | Training partition Dataset. |
| **`test`** | — | Testing partition Dataset. |
