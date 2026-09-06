# `Split`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`
**Source**: [`strata/model_selection/kfold.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/kfold.mojo)

```mojo
struct Split(Movable)
```

```mojo
from strata.model_selection import Split
```

**Pair of train and validation sample indices for a cross-validation fold.**

---

## Constructors

```mojo
def __init__(out self, var train_indices: List[Int], var val_indices: List[Int])
```

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Split.copy()`](#copy) | Returns a deep copy of this split pair. |

---

## Method Details

### `Split.copy()`

```mojo
def copy(self) -> Self
```
Returns a deep copy of this split pair.

**Returns**: `Self`

---
