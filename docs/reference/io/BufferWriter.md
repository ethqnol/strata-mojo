# `BufferWriter`

**Module**: [`strata.io`](index.md) &bull; **Kind**: `struct`
**Source**: [`strata/io/serializer.mojo`](file:////home/ewu/Code/Strata/strata/io/serializer.mojo)

```mojo
struct BufferWriter
```

```mojo
from strata.io import BufferWriter
```

**Byte buffer writer with endian-safe scalar and matrix serialization.**

---

## Constructors

```mojo
def __init__(out self)
```

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`BufferWriter.write_byte()`](#write_byte) | — |
| [`BufferWriter.write_bool()`](#write_bool) | — |
| [`BufferWriter.write_int()`](#write_int) | — |
| [`BufferWriter.write_float64()`](#write_float64) | — |
| [`BufferWriter.write_float32()`](#write_float32) | — |
| [`BufferWriter.write_string()`](#write_string) | — |
| [`BufferWriter.write_int_list()`](#write_int_list) | — |
| [`BufferWriter.write_float64_list()`](#write_float64_list) | — |
| [`BufferWriter.write_float_list()`](#write_float_list) | — |
| [`BufferWriter.write_matrix()`](#write_matrix) | — |
| [`BufferWriter.get_bytes()`](#get_bytes) | — |
| [`BufferWriter.save_to_file()`](#save_to_file) | — |

---

## Method Details

### `BufferWriter.write_byte()`

```mojo
def write_byte(mut self, val: UInt8)
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `UInt8` | — |

---

### `BufferWriter.write_bool()`

```mojo
def write_bool(mut self, val: Bool)
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `Bool` | — |

---

### `BufferWriter.write_int()`

```mojo
def write_int(mut self, val: Int)
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `Int` | — |

---

### `BufferWriter.write_float64()`

```mojo
def write_float64(mut self, val: Float64)
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `Float64` | — |

---

### `BufferWriter.write_float32()`

```mojo
def write_float32(mut self, val: Float32)
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `Float32` | — |

---

### `BufferWriter.write_string()`

```mojo
def write_string(mut self, val: String)
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `String` | — |

---

### `BufferWriter.write_int_list()`

```mojo
def write_int_list(mut self, val: List[Int])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `List[Int]` | — |

---

### `BufferWriter.write_float64_list()`

```mojo
def write_float64_list(mut self, val: List[Float64])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `List[Float64]` | — |

---

### `BufferWriter.write_float_list()`

```mojo
def write_float_list[dtype: DType](mut self, val: List[Scalar[dtype]])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `List[Scalar[dtype]]` | — |

---

### `BufferWriter.write_matrix()`

```mojo
def write_matrix[dtype: DType](mut self, val: Matrix[dtype])
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val`** | `Matrix[dtype]` | — |

---

### `BufferWriter.get_bytes()`

```mojo
def get_bytes(self) -> List[UInt8]
```
**Returns**: `List[UInt8]`

---

### `BufferWriter.save_to_file()`

```mojo
def save_to_file(self, path: String)
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`path`** | `String` | — |

---
