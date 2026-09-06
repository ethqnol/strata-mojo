# `TimeSeriesSplit`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`
**Source**: [`strata/model_selection/time_series_split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/time_series_split.mojo)

```mojo
struct TimeSeriesSplit(Movable)
```

```mojo
from strata.model_selection import TimeSeriesSplit
```

**Time Series cross-validator.**

Provides train/test indices to split time series data samples that are
observed at fixed time intervals. In each split, test indices must be
higher than before, so shuffling in cross-validator is inappropriate.
Successive training sets are supersets of those that come before them.

---

## Constructors

```mojo
def __init__(out self, n_splits: Int = 5, max_train_size: Int = 0, test_size: Int = 0, gap: Int = 0)
```

Initializes the TimeSeriesSplit cross-validator.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_splits`** | `Int` | Number of splits (must be at least 2). |
| **`max_train_size`** | `Int` | Maximum size for a single training set, or 0 for an expanding window with no upper bound. |
| **`test_size`** | `Int` | Size used for each test set, or 0 to default to n_samples // (n_splits + 1). |
| **`gap`** | `Int` | Number of samples to exclude between the end of each training set and the start of the following test set. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`TimeSeriesSplit.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |
| [`TimeSeriesSplit.split()`](#split) | Generates indices to split time-ordered data into train and test sets. |

---

## Method Details

### `TimeSeriesSplit.get_n_splits()`

```mojo
def get_n_splits(self) -> Int
```
Returns the number of splitting iterations in the cross-validator.

**Returns**: `Int`

---

### `TimeSeriesSplit.split()`

```mojo
def split(self, n_samples: Int) -> List[Split]
def split[dtype: DType](self, X: Matrix[dtype]) -> List[Split]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Generates indices to split time-ordered data into train and test sets.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`n_samples`** | `Int` | — |
| **`X`** | `Matrix[dtype]` | Feature matrix. |

**Returns**: `List[Split]` — List of Split objects containing train and validation indices.

---
