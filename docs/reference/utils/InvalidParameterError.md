# `InvalidParameterError`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`
**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)

```mojo
struct InvalidParameterError(Copyable, Movable, Writable)
```

```mojo
from strata.utils import InvalidParameterError
```

**Exception raised when an invalid hyperparameter value is supplied.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`InvalidParameterError.error()`](#error) | Create a formatted InvalidParameterError message. |
| [`InvalidParameterError.write_to()`](#write_to) | — |

---

## Method Details

### `InvalidParameterError.error()`

```mojo
def error(param_name: String, reason: String) -> Error
```
Create a formatted InvalidParameterError message.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`param_name`** | `String` | — |
| **`reason`** | `String` | — |

**Returns**: `Error`

---

### `InvalidParameterError.write_to()`

```mojo
def write_to(self, mut writer: Some[Writer])
```
---
