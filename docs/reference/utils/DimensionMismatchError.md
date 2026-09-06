# `DimensionMismatchError`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`
**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)

```mojo
struct DimensionMismatchError(Copyable, Movable, Writable)
```

```mojo
from strata.utils import DimensionMismatchError
```

**Exception raised when input matrix/vector dimensions do not match requirements.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`DimensionMismatchError.error()`](#error) | Create a formatted DimensionMismatchError message. |
| [`DimensionMismatchError.write_to()`](#write_to) | — |

---

## Method Details

### `DimensionMismatchError.error()`

```mojo
def error(expected: String, actual: String, context: String = "") -> Error
```
Create a formatted DimensionMismatchError message.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`expected`** | `String` | — |
| **`actual`** | `String` | — |
| **`context`** | `String` | — |

**Returns**: `Error`

---

### `DimensionMismatchError.write_to()`

```mojo
def write_to(self, mut writer: Some[Writer])
```
---
