# `PRNG`

**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/utils/random.mojo`](file:////home/ewu/Code/Strata/strata/utils/random.mojo)

```mojo
struct PRNG(Copyable, Movable)
```

```mojo
from strata.utils import PRNG
```

**64-bit SplitMix64 pseudo-random number generator with unbiased range generation.**

---

## Constructors

```mojo
def __init__(out self, seed: Int = 42)
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`seed`** | `Int` | — |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`PRNG.next_u64()`](#next_u64) | — |
| [`PRNG.next_int()`](#next_int) | Returns a pseudo-random integer in [0, upper_bound). |
| [`PRNG.permutation()`](#permutation) | Generates a pseudo-random permutation of range(0, n) using Fisher-Yates shuffle. |
| [`PRNG.shuffle()`](#shuffle) | In-place Fisher-Yates shuffle on a List. |

---

## Method Details

### `PRNG.next_u64()`

```mojo
def next_u64(mut self) -> UInt64
```
**Returns**: `UInt64`

---

### `PRNG.next_int()`

```mojo
def next_int(mut self, upper_bound: Int) -> Int
```
Returns a pseudo-random integer in [0, upper_bound).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`upper_bound`** | `Int` | — |

**Returns**: `Int`

---

### `PRNG.permutation()`

```mojo
def permutation(n: Int, seed: Int = 42) -> List[Int]
```
Generates a pseudo-random permutation of range(0, n) using Fisher-Yates shuffle.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`n`** | `Int` | — |
| **`seed`** | `Int` | — |

**Returns**: `List[Int]`

---

### `PRNG.shuffle()`

```mojo
def shuffle[T: Deinitable & Copyable](mut list: List[T], seed: Int = 42)
```
In-place Fisher-Yates shuffle on a List.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`seed`** | `Int` | — |

---
