# `train_test_split`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `function`  
**Source**: [`strata/model_selection/split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/split.mojo)

```mojo
def train_test_split[dtype: DType = DType.float64](X: Matrix[dtype], y: List[Scalar[dtype]], test_size: Float64 = 0.25, shuffle: Bool = True, seed: Int = 42) -> DatasetSplit[dtype, dtype]
```

```mojo
from strata.model_selection import train_test_split
```

**Split feature matrix and target list into train and test partitions.**

**Returns**: `DatasetSplit[dtype, dtype]` — DatasetSplit: Container holding partitioned training and testing datasets.
