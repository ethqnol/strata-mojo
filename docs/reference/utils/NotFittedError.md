# `NotFittedError`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`
**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)

```mojo
struct NotFittedError(Copyable, Movable, Writable)
```

```mojo
from strata.utils import NotFittedError
```

**Exception raised when an estimator is used before calling `fit`.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`NotFittedError.error()`](#error) | Create a formatted NotFittedError message. |
| [`NotFittedError.write_to()`](#write_to) | — |

---

## Method Details

### `NotFittedError.error()`

```mojo
def error(estimator_name: String, msg: String = "") -> Error
```
Create a formatted NotFittedError message.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`estimator_name`** | `String` | — |
| **`msg`** | `String` | — |

**Returns**: `Error`

---

### `NotFittedError.write_to()`

```mojo
def write_to(self, mut writer: Some[Writer])
```
---
