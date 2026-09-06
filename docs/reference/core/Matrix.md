# `Matrix`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `ArrayLike, Copyable, Movable, Writable`
**Source**: [`strata/core/matrix.mojo`](file:////home/ewu/Code/Strata/strata/core/matrix.mojo)

```mojo
struct Matrix[dtype: DType = DType.float64](ArrayLike, Copyable, Movable, Writable)
```

```mojo
from strata.core import Matrix
```

**Dense 2D row-major matrix container with striding and view support.**

Provides contiguous buffer allocation, SIMD-compatible row-major layout,
slicing, element-wise arithmetic, and BLAS/LAPACK interop.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`dtype`** | Numerical data type of matrix elements. Default DType.float64. |

---

## Constructors

```mojo
def __init__(out self, rows: Int, cols: Int, fill: Scalar[Self.dtype] = 0)
```

Initialize a Matrix of shape (rows, cols) filled with a constant value.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`rows`** | `Int` | Number of rows ($N >= 0$). |
| **`cols`** | `Int` | Number of columns ($D >= 0$). |
| **`fill`** | `Scalar[Self.dtype]` | Constant scalar value to fill the buffer. Default 0. |

```mojo
def __init__(out self, rows: Int, cols: Int, var data: List[Scalar[Self.dtype]])
```

Initialize a Matrix from an existing 1D buffer.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`rows`** | `Int` | Number of rows ($N >= 0$). |
| **`cols`** | `Int` | Number of columns ($D >= 0$). |
| **`data`** | — | Flat row-major element buffer of length `rows * cols`. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`rows`** | Number of matrix rows ($N$). |
| **`cols`** | Number of matrix columns ($D$). |
| **`data`** | Flat 1D buffer of matrix elements in row-major order. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Matrix.zeros()`](#zeros) | Create a zero-filled Matrix of shape (rows, cols). |
| [`Matrix.ones()`](#ones) | Create a one-filled Matrix of shape (rows, cols). |
| [`Matrix.eye()`](#eye) | Create an identity Matrix of shape (n, n). |
| [`Matrix.from_numpy()`](#from_numpy) | — |
| [`Matrix.to_numpy()`](#to_numpy) | — |
| [`Matrix.num_rows()`](#num_rows) | — |
| [`Matrix.num_cols()`](#num_cols) | — |
| [`Matrix.num_elements()`](#num_elements) | — |
| [`Matrix.shape()`](#shape) | — |
| [`Matrix.view()`](#view) | — |
| [`Matrix.slice_rows()`](#slice_rows) | — |
| [`Matrix.slice_cols()`](#slice_cols) | — |
| [`Matrix.slice_2d()`](#slice_2d) | — |
| [`Matrix.row()`](#row) | — |
| [`Matrix.col()`](#col) | — |
| [`Matrix.cast()`](#cast) | Promotes or converts the Matrix elements to target_dtype. |
| [`Matrix.transpose()`](#transpose) | — |
| [`Matrix.select_columns()`](#select_columns) | Extracts an $N \times K$ submatrix containing only the specified column indices in order. |
| [`Matrix.select_rows()`](#select_rows) | Extracts an $M \times D$ submatrix containing only the specified row indices in order. |
| [`Matrix.dot()`](#dot) | — |
| [`Matrix.dot_vec()`](#dot_vec) | — |
| [`Matrix.mean_along_axis_0()`](#mean_along_axis_0) | — |
| [`Matrix.std_along_axis_0()`](#std_along_axis_0) | — |
| [`Matrix.write_to()`](#write_to) | — |
| [`Matrix.hstack()`](#hstack) | Horizontally stacks a list of matrices with matching row counts. |
| [`Matrix.vstack()`](#vstack) | Vertically stacks a list of matrices with matching column counts. |

---

## Method Details

### `Matrix.zeros()`

```mojo
def zeros(rows: Int, cols: Int) -> Self
```
Create a zero-filled Matrix of shape (rows, cols).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`rows`** | `Int` | — |
| **`cols`** | `Int` | — |

**Returns**: `Self`

---

### `Matrix.ones()`

```mojo
def ones(rows: Int, cols: Int) -> Self
```
Create a one-filled Matrix of shape (rows, cols).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`rows`** | `Int` | — |
| **`cols`** | `Int` | — |

**Returns**: `Self`

---

### `Matrix.eye()`

```mojo
def eye(n: Int) -> Self
```
Create an identity Matrix of shape (n, n).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`n`** | `Int` | — |

**Returns**: `Self`

---

### `Matrix.from_numpy()`

```mojo
def from_numpy(np_arr: PythonObject) -> Self
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`np_arr`** | `PythonObject` | — |

**Returns**: `Self`

---

### `Matrix.to_numpy()`

```mojo
def to_numpy(self) -> PythonObject
```
**Returns**: `PythonObject`

---

### `Matrix.num_rows()`

```mojo
def num_rows(self) -> Int
```
**Returns**: `Int`

---

### `Matrix.num_cols()`

```mojo
def num_cols(self) -> Int
```
**Returns**: `Int`

---

### `Matrix.num_elements()`

```mojo
def num_elements(self) -> Int
```
**Returns**: `Int`

---

### `Matrix.shape()`

```mojo
def shape(self) -> Tuple[Int, Int]
```
**Returns**: `Tuple[Int, Int]`

---

### `Matrix.view()`

```mojo
def view(ref self) -> MatrixView[Self.dtype, origin_of(self.data)]
```
**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`

---

### `Matrix.slice_rows()`

```mojo
def slice_rows(ref self, start_row: Int, end_row: Int) -> MatrixView[Self.dtype, origin_of(self.data)]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`start_row`** | `Int` | — |
| **`end_row`** | `Int` | — |

**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`

---

### `Matrix.slice_cols()`

```mojo
def slice_cols(ref self, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, origin_of(self.data)]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`start_col`** | `Int` | — |
| **`end_col`** | `Int` | — |

**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`

---

### `Matrix.slice_2d()`

```mojo
def slice_2d(ref self, start_row: Int, end_row: Int, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, origin_of(self.data)]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`start_row`** | `Int` | — |
| **`end_row`** | `Int` | — |
| **`start_col`** | `Int` | — |
| **`end_col`** | `Int` | — |

**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`

---

### `Matrix.row()`

```mojo
def row(self, r: Int) -> List[Scalar[Self.dtype]]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`r`** | `Int` | — |

**Returns**: `List[Scalar[Self.dtype]]`

---

### `Matrix.col()`

```mojo
def col(self, c: Int) -> List[Scalar[Self.dtype]]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`c`** | `Int` | — |

**Returns**: `List[Scalar[Self.dtype]]`

---

### `Matrix.cast()`

```mojo
def cast[target_dtype: DType](self) -> Matrix[target_dtype]
```
Promotes or converts the Matrix elements to target_dtype.

**Returns**: `Matrix[target_dtype]`

---

### `Matrix.transpose()`

```mojo
def transpose(self) -> Self
```
**Returns**: `Self`

---

### `Matrix.select_columns()`

```mojo
def select_columns(self, indices: List[Int]) -> Self
```
Extracts an $N \times K$ submatrix containing only the specified column indices in order.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`indices`** | `List[Int]` | — |

**Returns**: `Self` — Matrix[dtype]: Submatrix of shape (self.rows, len(indices)).

---

### `Matrix.select_rows()`

```mojo
def select_rows(self, indices: List[Int]) -> Self
```
Extracts an $M \times D$ submatrix containing only the specified row indices in order.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`indices`** | `List[Int]` | — |

**Returns**: `Self` — Matrix[dtype]: Submatrix of shape (len(indices), self.cols).

---

### `Matrix.dot()`

```mojo
def dot(self, other: Self) -> Self
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`other`** | `Self` | — |

**Returns**: `Self`

---

### `Matrix.dot_vec()`

```mojo
def dot_vec(self, vec: List[Scalar[Self.dtype]]) -> List[Scalar[Self.dtype]]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`vec`** | `List[Scalar[Self.dtype]]` | — |

**Returns**: `List[Scalar[Self.dtype]]`

---

### `Matrix.mean_along_axis_0()`

```mojo
def mean_along_axis_0(self) -> List[Scalar[Self.dtype]]
```
**Returns**: `List[Scalar[Self.dtype]]`

---

### `Matrix.std_along_axis_0()`

```mojo
def std_along_axis_0(self, means: List[Scalar[Self.dtype]]) -> List[Scalar[Self.dtype]]
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`means`** | `List[Scalar[Self.dtype]]` | — |

**Returns**: `List[Scalar[Self.dtype]]`

---

### `Matrix.write_to()`

```mojo
def write_to(self, mut writer: Some[Writer])
```
---

### `Matrix.hstack()`

```mojo
def hstack[dtype: DType = DType.float64](matrices: List[Matrix[dtype]]) -> Matrix[dtype]
def hstack[dtype: DType = DType.float64](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Horizontally stacks a list of matrices with matching row counts.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`matrices`** | `List[Matrix[dtype]]` | — |
| **`A`** | `Matrix[dtype]` | — |
| **`B`** | `Matrix[dtype]` | — |

**Returns**: `Matrix[dtype]` — Matrix[dtype]: Concatenated matrix of shape (rows, sum(cols)).

---

### `Matrix.vstack()`

```mojo
def vstack[dtype: DType = DType.float64](matrices: List[Matrix[dtype]]) -> Matrix[dtype]
def vstack[dtype: DType = DType.float64](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]
```
> **Overload Note**: This method supports multiple overloaded call signatures.

Vertically stacks a list of matrices with matching column counts.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`matrices`** | `List[Matrix[dtype]]` | — |
| **`A`** | `Matrix[dtype]` | — |
| **`B`** | `Matrix[dtype]` | — |

**Returns**: `Matrix[dtype]` — Matrix[dtype]: Concatenated matrix of shape (sum(rows), cols).
---

## Example

```mojo
from strata.core import Matrix

var A = Matrix[DType.float64](2, 3, fill=1.0)
var B = Matrix[DType.float64].eye(3)
var C = A.dot(B)
```
