# `Dataset`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`
**Source**: [`strata/core/dataset.mojo`](file:////home/ewu/Code/Strata/strata/core/dataset.mojo)

```mojo
struct Dataset[feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.core import Dataset
```

**Machine learning dataset container pairing a feature matrix with targets.**

Encapsulates 2D feature observations, 1D target labels/values, and optional
feature/target name metadata.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`feat_dtype`** | Data type of the feature matrix (default: Float64). |
| **`target_dtype`** | Data type of the target values (default: Float64). |

---

## Constructors

```mojo
def __init__(out self, var records: Matrix[Self.feat_dtype], var targets: List[Scalar[Self.target_dtype]])
```

Initialize a Dataset from records and targets.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`records`** | — | Feature matrix of shape $(N, D)$. |
| **`targets`** | — | Target vector of length $N$. |

```mojo
def __init__(out self, var records: Matrix[Self.feat_dtype], var targets: List[Scalar[Self.target_dtype]], var feature_names: List[String], var target_names: List[String])
```

```mojo
def __init__(out self, *, deinit move: Self)
```

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`records`** | Feature matrix of shape $(N, D)$. |
| **`targets`** | Target label/value vector of length $N$. |
| **`feature_names`** | List of feature names of length $D$. |
| **`target_names`** | List of class or target variable names. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Dataset.n_samples()`](#n_samples) | — |
| [`Dataset.n_features()`](#n_features) | — |
| [`Dataset.split_with_ratio()`](#split_with_ratio) | — |

---

## Method Details

### `Dataset.n_samples()`

```mojo
def n_samples(self) -> Int
```
**Returns**: `Int`

---

### `Dataset.n_features()`

```mojo
def n_features(self) -> Int
```
**Returns**: `Int`

---

### `Dataset.split_with_ratio()`

```mojo
def split_with_ratio(self, ratio: Float64 = 0.25, shuffle: Bool = True, seed: Int = 42) -> DatasetSplit[Self.feat_dtype, Self.target_dtype]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`ratio`** | `Float64` | — |
| **`shuffle`** | `Bool` | — |
| **`seed`** | `Int` | — |

**Returns**: `DatasetSplit[Self.feat_dtype, Self.target_dtype]`
---

## Example

```mojo
from strata.core import Matrix, Dataset

var X = Matrix[DType.float64](100, 4, fill=1.0)
var y = List[Scalar[DType.float64]](capacity=100)
var ds = Dataset(X^, y^)
var split = ds.split_with_ratio(ratio=0.2)
```
