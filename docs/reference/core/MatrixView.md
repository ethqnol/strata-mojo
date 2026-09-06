# `MatrixView`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `ArrayLike, Copyable, Movable`
**Source**: [`strata/core/view.mojo`](file:////home/ewu/Code/Strata/strata/core/view.mojo)

```mojo
struct MatrixView[dtype: DType, origin: Origin](ArrayLike, Copyable, Movable)
```

```mojo
from strata.core import MatrixView
```

**Non-owning 2D view over a contiguous or strided matrix memory buffer.**

---

## Constructors

```mojo
def __init__(out self, ptr: Pointer[Scalar[Self.dtype], Self.origin], rows: Int, cols: Int, row_stride: Int, col_stride: Int = 1)
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`ptr`** | `Pointer[Scalar[Self.dtype], Self.origin]` | — |
| **`rows`** | `Int` | — |
| **`cols`** | `Int` | — |
| **`row_stride`** | `Int` | — |
| **`col_stride`** | `Int` | — |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`MatrixView.num_rows()`](#num_rows) | — |
| [`MatrixView.num_cols()`](#num_cols) | — |
| [`MatrixView.num_elements()`](#num_elements) | — |
| [`MatrixView.shape()`](#shape) | — |
| [`MatrixView.slice_rows()`](#slice_rows) | — |
| [`MatrixView.slice_cols()`](#slice_cols) | — |
| [`MatrixView.slice_2d()`](#slice_2d) | — |
| [`MatrixView.to_matrix()`](#to_matrix) | — |

---

## Method Details

### `MatrixView.num_rows()`

```mojo
def num_rows(self) -> Int
```
**Returns**: `Int`

---

### `MatrixView.num_cols()`

```mojo
def num_cols(self) -> Int
```
**Returns**: `Int`

---

### `MatrixView.num_elements()`

```mojo
def num_elements(self) -> Int
```
**Returns**: `Int`

---

### `MatrixView.shape()`

```mojo
def shape(self) -> Tuple[Int, Int]
```
**Returns**: `Tuple[Int, Int]`

---

### `MatrixView.slice_rows()`

```mojo
def slice_rows(self, start_row: Int, end_row: Int) -> MatrixView[Self.dtype, Self.origin]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`start_row`** | `Int` | — |
| **`end_row`** | `Int` | — |

**Returns**: `MatrixView[Self.dtype, Self.origin]`

---

### `MatrixView.slice_cols()`

```mojo
def slice_cols(self, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, Self.origin]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`start_col`** | `Int` | — |
| **`end_col`** | `Int` | — |

**Returns**: `MatrixView[Self.dtype, Self.origin]`

---

### `MatrixView.slice_2d()`

```mojo
def slice_2d(self, start_row: Int, end_row: Int, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, Self.origin]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`start_row`** | `Int` | — |
| **`end_row`** | `Int` | — |
| **`start_col`** | `Int` | — |
| **`end_col`** | `Int` | — |

**Returns**: `MatrixView[Self.dtype, Self.origin]`

---

### `MatrixView.to_matrix()`

```mojo
def to_matrix(self) -> Matrix[Self.dtype]
```
**Returns**: `Matrix[Self.dtype]`

---
