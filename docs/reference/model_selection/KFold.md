# `KFold`

**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`
**Source**: [`strata/model_selection/kfold.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/kfold.mojo)

```mojo
struct KFold(Movable)
```

```mojo
from strata.model_selection import KFold
```

**K-Fold cross-validator.**

Provides train/validation indices to split data into $K$ consecutive or shuffled folds.
Each fold is used once as a validation set while the remaining $K-1$ folds form the
training set.

---

## Constructors

```mojo
def __init__(out self, n_splits: Int = 5, shuffle: Bool = False, random_state: Int = 42)
```

Initialize the KFold splitter.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`n_splits`** | `Int` | Number of folds (must be at least 2). Default 5. |
| **`shuffle`** | `Bool` | Whether to shuffle the data before splitting. Default False. |
| **`random_state`** | `Int` | PRNG seed when shuffle is True. Default 42. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`KFold.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |
| [`KFold.split()`](#split) | Generates indices to split data into training and test sets. |

---

## Method Details

### `KFold.get_n_splits()`

```mojo
def get_n_splits(self) -> Int
```
Returns the number of splitting iterations in the cross-validator.

**Returns**: `Int`

---

### `KFold.split()`

```mojo
def split(self, n_samples: Int) -> List[Split]
def split[dtype: DType](self, X: Matrix[dtype]) -> List[Split]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Generates indices to split data into training and test sets.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`n_samples`** | `Int` | — |
| **`X`** | `Matrix[dtype]` | Feature matrix. |

**Returns**: `List[Split]` — List of Split objects containing train and validation indices.
---

## Example

```mojo
from strata.model_selection import KFold

var kf = KFold(n_splits=5, shuffle=True, random_state=42)
var folds = kf.split(n_samples=100)
```
