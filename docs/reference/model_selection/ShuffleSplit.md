# `ShuffleSplit`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  
**Source**: [`strata/model_selection/shuffle_split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/shuffle_split.mojo)

```mojo
struct ShuffleSplit(Movable)
```

```mojo
from strata.model_selection import ShuffleSplit
```

**Random permutation cross-validator.**

Yields indices to split data into training and test sets. Each split is an
independent random permutation of the samples, so successive test sets may
overlap. Sizes are expressed as proportions of the total sample count.

---

## Constructors

```mojo
def __init__(out self, n_splits: Int = 10, test_size: Float64 = 0.1, train_size: Float64 = 0.0, random_state: Int = 42)
```

Initializes the ShuffleSplit cross-validator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_splits`** | `Int` | Number of re-shuffling and splitting iterations. |
| **`test_size`** | `Float64` | Proportion of samples held out for testing, in (0, 1). |
| **`train_size`** | `Float64` | Proportion of samples used for training, in (0, 1), or 0.0 to use the complement of test_size. |
| **`random_state`** | `Int` | Base seed controlling the permutation of each split. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`ShuffleSplit.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |
| [`ShuffleSplit.split()`](#split) | Generates randomly permuted train and test indices for each split. |

---

## Method Details

### `ShuffleSplit.get_n_splits()`

```mojo
def get_n_splits(self) -> Int
```
Returns the number of splitting iterations in the cross-validator.

**Returns**: `Int`

---

### `ShuffleSplit.split()`

```mojo
def split(self, n_samples: Int) -> List[Split]
def split[dtype: DType](self, X: Matrix[dtype]) -> List[Split]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Generates randomly permuted train and test indices for each split.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`n_samples`** | `Int` | — |
| **`X`** | `Matrix[dtype]` | Feature matrix. |

**Returns**: `List[Split]` — List of Split objects containing train and validation indices.

---
