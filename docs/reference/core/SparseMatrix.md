# `SparseMatrix`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `trait`  
**Source**: [`strata/core/sparse.mojo`](file:////home/ewu/Code/Strata/strata/core/sparse.mojo)

```mojo
trait SparseMatrix
```

```mojo
from strata.core import SparseMatrix
```

**Base interface trait for 2D sparse matrix representations.**

Provides common dimension queries (`num_rows`, `num_cols`) and structural
sparsity counts (`nnz`) for Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) formats.

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`SparseMatrix.num_rows()`](#num_rows) | — |
| [`SparseMatrix.num_cols()`](#num_cols) | — |
| [`SparseMatrix.nnz()`](#nnz) | — |

---

## Method Details

### `SparseMatrix.num_rows()`

```mojo
def num_rows(self) -> Int
```
**Returns**: `Int`

---

### `SparseMatrix.num_cols()`

```mojo
def num_cols(self) -> Int
```
**Returns**: `Int`

---

### `SparseMatrix.nnz()`

```mojo
def nnz(self) -> Int
```
**Returns**: `Int`

---
