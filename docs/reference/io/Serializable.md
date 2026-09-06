# `Serializable`

**Module**: [`strata.io`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Copyable, Movable`
**Source**: [`strata/io/serializer.mojo`](file:////home/ewu/Code/Strata/strata/io/serializer.mojo)

```mojo
trait Serializable(Copyable, Movable)
```

```mojo
from strata.io import Serializable
```

**Trait for models and transformers supporting binary serialization.**

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Serializable.serialize()`](#serialize) | Serializes internal parameters and fitted state into BufferWriter. |
| [`Serializable.deserialize()`](#deserialize) | Deserializes a model instance from BufferReader. |
| [`Serializable.dump()`](#dump) | Serializes a fitted estimator or transformer to disk. |
| [`Serializable.dumps()`](#dumps) | Serializes a fitted estimator or transformer into an in-memory byte buffer. |
| [`Serializable.load()`](#load) | Deserializes a fitted estimator or transformer from disk. |
| [`Serializable.loads()`](#loads) | Deserializes a fitted estimator or transformer from an in-memory byte buffer. |

---

## Method Details

### `Serializable.serialize()`

```mojo
def serialize(self, mut writer: BufferWriter)
```
Serializes internal parameters and fitted state into BufferWriter.

---

### `Serializable.deserialize()`

```mojo
def deserialize(mut reader: BufferReader) -> Self
```
Deserializes a model instance from BufferReader.

**Returns**: `Self`

---

### `Serializable.dump()`

```mojo
def dump[T: Serializable](model: T, path: String)
```
Serializes a fitted estimator or transformer to disk.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`model`** | `T` | — |
| **`path`** | `String` | — |

---

### `Serializable.dumps()`

```mojo
def dumps[T: Serializable](model: T) -> List[UInt8]
```
Serializes a fitted estimator or transformer into an in-memory byte buffer.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`model`** | `T` | — |

**Returns**: `List[UInt8]` — List[UInt8]: Packed binary byte buffer.

---

### `Serializable.load()`

```mojo
def load[T: Serializable](path: String) -> T
```
Deserializes a fitted estimator or transformer from disk.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`T`** | — | Concrete model struct type. |

**Returns**: `T` — T: Restored model instance with fitted state intact.

---

### `Serializable.loads()`

```mojo
def loads[T: Serializable](bytes: List[UInt8]) -> T
```
Deserializes a fitted estimator or transformer from an in-memory byte buffer.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`T`** | — | Concrete model struct type. |

**Returns**: `T` — T: Restored model instance with fitted state intact.

---
