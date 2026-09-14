# `BufferReader`

**Module**: [`strata.io`](index.md) &bull; **Kind**: `struct`  
**Source**: [`strata/io/serializer.mojo`](file:////home/ewu/Code/Strata/strata/io/serializer.mojo)

```mojo
struct BufferReader
```

```mojo
from strata.io import BufferReader
```

**Byte buffer reader with bounds checking and scalar/matrix deserialization.**

---

## Constructors

```mojo
def __init__(out self, buffer: List[UInt8])
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`buffer`** | `List[UInt8]` | — |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`BufferReader.from_file()`](#from_file) | — |
| [`BufferReader.read_byte()`](#read_byte) | — |
| [`BufferReader.read_bool()`](#read_bool) | — |
| [`BufferReader.read_int()`](#read_int) | — |
| [`BufferReader.read_float64()`](#read_float64) | — |
| [`BufferReader.read_float32()`](#read_float32) | — |
| [`BufferReader.read_string()`](#read_string) | — |
| [`BufferReader.read_int_list()`](#read_int_list) | — |
| [`BufferReader.read_float64_list()`](#read_float64_list) | — |
| [`BufferReader.read_float_list()`](#read_float_list) | — |
| [`BufferReader.read_matrix()`](#read_matrix) | — |
| [`BufferReader.write_header()`](#write_header) | Writes protocol magic header, version tag, and model type identifier. |
| [`BufferReader.check_header()`](#check_header) | Validates protocol magic header, version tag, and model type identifier. |

---

## Method Details

### `BufferReader.from_file()`

```mojo
def from_file(path: String) -> Self
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`path`** | `String` | — |

**Returns**: `Self`

---

### `BufferReader.read_byte()`

```mojo
def read_byte(mut self) -> UInt8
```
**Returns**: `UInt8`

---

### `BufferReader.read_bool()`

```mojo
def read_bool(mut self) -> Bool
```
**Returns**: `Bool`

---

### `BufferReader.read_int()`

```mojo
def read_int(mut self) -> Int
```
**Returns**: `Int`

---

### `BufferReader.read_float64()`

```mojo
def read_float64(mut self) -> Float64
```
**Returns**: `Float64`

---

### `BufferReader.read_float32()`

```mojo
def read_float32(mut self) -> Float32
```
**Returns**: `Float32`

---

### `BufferReader.read_string()`

```mojo
def read_string(mut self) -> String
```
**Returns**: `String`

---

### `BufferReader.read_int_list()`

```mojo
def read_int_list(mut self) -> List[Int]
```
**Returns**: `List[Int]`

---

### `BufferReader.read_float64_list()`

```mojo
def read_float64_list(mut self) -> List[Float64]
```
**Returns**: `List[Float64]`

---

### `BufferReader.read_float_list()`

```mojo
def read_float_list[dtype: DType](mut self) -> List[Scalar[dtype]]
```
**Returns**: `List[Scalar[dtype]]`

---

### `BufferReader.read_matrix()`

```mojo
def read_matrix[dtype: DType](mut self) -> Matrix[dtype]
```
**Returns**: `Matrix[dtype]`

---

### `BufferReader.write_header()`

```mojo
def write_header(mut writer: BufferWriter, type_name: String)
```
Writes protocol magic header, version tag, and model type identifier.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`type_name`** | `String` | — |

---

### `BufferReader.check_header()`

```mojo
def check_header(mut reader: BufferReader, expected_type: String)
```
Validates protocol magic header, version tag, and model type identifier.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`expected_type`** | `String` | — |

---
