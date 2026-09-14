# `EigResult`

**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/core/linalg.mojo`](file:////home/ewu/Code/Strata/strata/core/linalg.mojo)

```mojo
struct EigResult[dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.core import EigResult
```

**Result of Symmetric Eigenvalue Decomposition ($A V = V \Lambda$).**

---

## Constructors

```mojo
def __init__(out self, var eigenvalues: List[Scalar[Self.dtype]], var eigenvectors: Matrix[Self.dtype])
```

Initialize an EigResult container.

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`eigenvalues`** | Real eigenvalues vector of length $N$ in ascending order. |
| **`eigenvectors`** | Eigenvector matrix of shape $(N, N)$ with columns representing eigenvectors. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`EigResult.gemm()`](#gemm) | Compute dense matrix product $C = A B$. |
| [`EigResult.dense_dot_vec()`](#dense_dot_vec) | Dense matrix-vector product: y = A @ x + bias. |
| [`EigResult.svd()`](#svd) | Computes the Singular Value Decomposition (SVD) of a matrix A = U * S * Vt. |
| [`EigResult.qr()`](#qr) | Computes the QR Decomposition of matrix A = Q * R using LAPACK Householder reflectors (dgeqrf/dorgqr). |
| [`EigResult.cholesky()`](#cholesky) | Computes the Cholesky factorization of a symmetric positive-definite matrix A = L * L^T. |
| [`EigResult.lstsq()`](#lstsq) | Solve linear least-squares problem $\min_x \|A x - b\|_2$ using SVD. |
| [`EigResult.solve()`](#solve) | Solves a square linear system A * x = b using LU decomposition (dgesv/sgesv). |
| [`EigResult.solve_cholesky()`](#solve_cholesky) | Solves a symmetric positive definite linear system A * x = b using Cholesky (dposv/sposv). |
| [`EigResult.inv()`](#inv) | Computes the multiplicative inverse of a square matrix A using LU decomposition (dgetrf/dgetri). |
| [`EigResult.norm()`](#norm) | Computes the Frobenius matrix norm: ||A||_F = sqrt(sum(A_ij^2)). |
| [`EigResult.eigh()`](#eigh) | Computes the eigenvalues and eigenvectors of a real symmetric matrix. |

---

## Method Details

### `EigResult.gemm()`

```mojo
def gemm[dtype: DType = DType.float64](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]
```
Compute dense matrix product $C = A B$.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`B`** | `Matrix[dtype]` | — |

**Returns**: `Matrix[dtype]` — Matrix[dtype]: Output matrix product $C$ of shape $(M, N)$.

---

### `EigResult.dense_dot_vec()`

```mojo
def dense_dot_vec[dtype: DType = DType.float64](A: Matrix[dtype], x: List[Scalar[dtype]], bias: Scalar[dtype] = 0) -> List[Scalar[dtype]]
```
Dense matrix-vector product: y = A @ x + bias.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`x`** | `List[Scalar[dtype]]` | — |
| **`bias`** | `Scalar[dtype]` | — |

**Returns**: `List[Scalar[dtype]]`

---

### `EigResult.svd()`

```mojo
def svd[dtype: DType = DType.float64](A: Matrix[dtype], full_matrices: Bool = False) -> SVDResult[dtype]
```
Computes the Singular Value Decomposition (SVD) of a matrix A = U * S * Vt.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`full_matrices`** | `Bool` | — |

**Returns**: `SVDResult[dtype]`

---

### `EigResult.qr()`

```mojo
def qr[dtype: DType = DType.float64](A: Matrix[dtype]) -> QRResult[dtype]
```
Computes the QR Decomposition of matrix A = Q * R using LAPACK Householder reflectors (dgeqrf/dorgqr).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |

**Returns**: `QRResult[dtype]`

---

### `EigResult.cholesky()`

```mojo
def cholesky[dtype: DType = DType.float64](A: Matrix[dtype], lower: Bool = True) -> Matrix[dtype]
```
Computes the Cholesky factorization of a symmetric positive-definite matrix A = L * L^T.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`lower`** | `Bool` | — |

**Returns**: `Matrix[dtype]`

---

### `EigResult.lstsq()`

```mojo
def lstsq[dtype: DType = DType.float64](A: Matrix[dtype], b: List[Scalar[dtype]], rcond: Float64 = -1.0) -> List[Scalar[dtype]]
```
Solve linear least-squares problem $\min_x \|A x - b\|_2$ using SVD.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`b`** | `List[Scalar[dtype]]` | — |
| **`rcond`** | `Float64` | — |

**Returns**: `List[Scalar[dtype]]` — List[Scalar[dtype]]: Least-squares solution vector $x$ of length $N$.

---

### `EigResult.solve()`

```mojo
def solve[dtype: DType = DType.float64](A: Matrix[dtype], b: List[Scalar[dtype]]) -> List[Scalar[dtype]]
```
Solves a square linear system A * x = b using LU decomposition (dgesv/sgesv).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`b`** | `List[Scalar[dtype]]` | — |

**Returns**: `List[Scalar[dtype]]`

---

### `EigResult.solve_cholesky()`

```mojo
def solve_cholesky[dtype: DType = DType.float64](A: Matrix[dtype], b: List[Scalar[dtype]], lower: Bool = True) -> List[ Scalar[dtype] ]
```
Solves a symmetric positive definite linear system A * x = b using Cholesky (dposv/sposv).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`b`** | `List[Scalar[dtype]]` | — |
| **`lower`** | `Bool` | — |

**Returns**: `List[ Scalar[dtype] ]`

---

### `EigResult.inv()`

```mojo
def inv[dtype: DType = DType.float64](A: Matrix[dtype]) -> Matrix[dtype]
```
Computes the multiplicative inverse of a square matrix A using LU decomposition (dgetrf/dgetri).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |

**Returns**: `Matrix[dtype]`

---

### `EigResult.norm()`

```mojo
def norm[dtype: DType = DType.float64](A: Matrix[dtype], ord: String = "fro") -> Float64
```
Computes the Frobenius matrix norm: ||A||_F = sqrt(sum(A_ij^2)).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`ord`** | `String` | — |

**Returns**: `Float64`

---

### `EigResult.eigh()`

```mojo
def eigh[dtype: DType = DType.float64](A: Matrix[dtype], UPLO: String = "L") -> EigResult[dtype]
```
Computes the eigenvalues and eigenvectors of a real symmetric matrix.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`A`** | `Matrix[dtype]` | — |
| **`UPLO`** | `String` | — |

**Returns**: `EigResult[dtype]`

---
