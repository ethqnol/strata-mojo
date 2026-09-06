# `ConvergenceError`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`
**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)

```mojo
struct ConvergenceError(Copyable, Movable, Writable)
```

```mojo
from strata.utils import ConvergenceError
```

**Exception raised when iterative optimization fails to converge within max iterations.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`ConvergenceError.error()`](#error) | Create a formatted ConvergenceError message. |
| [`ConvergenceError.write_to()`](#write_to) | — |

---

## Method Details

### `ConvergenceError.error()`

```mojo
def error(estimator_name: String, max_iter: Int, loss: Float64 = 0.0) -> Error
```
Create a formatted ConvergenceError message.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`estimator_name`** | `String` | — |
| **`max_iter`** | `Int` | — |
| **`loss`** | `Float64` | — |

**Returns**: `Error`

---

### `ConvergenceError.write_to()`

```mojo
def write_to(self, mut writer: Some[Writer])
```
---
