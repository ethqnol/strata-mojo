from std.math import exp, tanh, pow
from ..core.matrix import Matrix
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


@always_inline
def _dot_rows[
    dtype: DType
](X1: Matrix[dtype], r1: Int, X2: Matrix[dtype], r2: Int) -> Scalar[dtype]:
    var d = X1.cols
    var p1 = X1.data.unsafe_ptr().unsafe_offset(r1 * d)
    var p2 = X2.data.unsafe_ptr().unsafe_offset(r2 * d)
    comptime simd_w = 4 if dtype == DType.float64 else 8

    var sum_simd = SIMD[dtype, simd_w](0)
    var j = 0
    while j + simd_w <= d:
        var v1 = p1.unsafe_offset(j).unsafe_load[width=simd_w]()
        var v2 = p2.unsafe_offset(j).unsafe_load[width=simd_w]()
        sum_simd = v1.fma(v2, sum_simd)
        j += simd_w

    var total: Scalar[dtype] = sum_simd.reduce_add()
    while j < d:
        total += p1.unsafe_offset(j)[] * p2.unsafe_offset(j)[]
        j += 1
    return total


@always_inline
def _sq_dist_rows[
    dtype: DType
](X1: Matrix[dtype], r1: Int, X2: Matrix[dtype], r2: Int) -> Scalar[dtype]:
    var d = X1.cols
    var p1 = X1.data.unsafe_ptr().unsafe_offset(r1 * d)
    var p2 = X2.data.unsafe_ptr().unsafe_offset(r2 * d)
    comptime simd_w = 4 if dtype == DType.float64 else 8

    var sum_simd = SIMD[dtype, simd_w](0)
    var j = 0
    while j + simd_w <= d:
        var diff = (
            p1.unsafe_offset(j).unsafe_load[width=simd_w]()
            - p2.unsafe_offset(j).unsafe_load[width=simd_w]()
        )
        sum_simd = diff.fma(diff, sum_simd)
        j += simd_w

    var total: Scalar[dtype] = sum_simd.reduce_add()
    while j < d:
        var diff = p1.unsafe_offset(j)[] - p2.unsafe_offset(j)[]
        total += diff * diff
        j += 1
    return total


@always_inline
def _dot_vec_row[
    dtype: DType
](v: List[Scalar[dtype]], X: Matrix[dtype], r: Int) -> Scalar[dtype]:
    var d = X.cols
    var p1 = v.unsafe_ptr()
    var p2 = X.data.unsafe_ptr().unsafe_offset(r * d)
    comptime simd_w = 4 if dtype == DType.float64 else 8

    var sum_simd = SIMD[dtype, simd_w](0)
    var j = 0
    while j + simd_w <= d:
        var v1 = p1.unsafe_offset(j).unsafe_load[width=simd_w]()
        var v2 = p2.unsafe_offset(j).unsafe_load[width=simd_w]()
        sum_simd = v1.fma(v2, sum_simd)
        j += simd_w

    var total: Scalar[dtype] = sum_simd.reduce_add()
    while j < d:
        total += p1.unsafe_offset(j)[] * p2.unsafe_offset(j)[]
        j += 1
    return total


@always_inline
def _sq_dist_vec_row[
    dtype: DType
](v: List[Scalar[dtype]], X: Matrix[dtype], r: Int) -> Scalar[dtype]:
    var d = X.cols
    var p1 = v.unsafe_ptr()
    var p2 = X.data.unsafe_ptr().unsafe_offset(r * d)
    comptime simd_w = 4 if dtype == DType.float64 else 8

    var sum_simd = SIMD[dtype, simd_w](0)
    var j = 0
    while j + simd_w <= d:
        var diff = (
            p1.unsafe_offset(j).unsafe_load[width=simd_w]()
            - p2.unsafe_offset(j).unsafe_load[width=simd_w]()
        )
        sum_simd = diff.fma(diff, sum_simd)
        j += simd_w

    var total: Scalar[dtype] = sum_simd.reduce_add()
    while j < d:
        var diff = p1.unsafe_offset(j)[] - p2.unsafe_offset(j)[]
        total += diff * diff
        j += 1
    return total


def compute_gamma[
    dtype: DType
](
    X: Matrix[dtype],
    gamma_str: String = "scale",
    gamma_val: Scalar[dtype] = -1.0,
) raises -> Scalar[dtype]:
    """Computes kernel scale parameter gamma based on dataset statistics.

    Parameters:
        dtype: Floating point precision type.

    Args:
        X: Matrix of shape (n_samples, n_features).
        gamma_str: Strategy ('scale' or 'auto'). Default 'scale'.
        gamma_val: Explicit numeric gamma. When > 0, overrides strategy.

    Returns:
        Scalar[dtype]: Resolved gamma parameter.

    Raises:
        InvalidParameterError: If gamma_str is unrecognized or gamma_val <= 0.
    """
    if gamma_val > 0.0:
        return gamma_val

    var d = X.cols
    if d == 0 or X.rows == 0:
        return Scalar[dtype](1.0)

    if gamma_str == "auto":
        return Scalar[dtype](1.0) / Scalar[dtype](d)

    if gamma_str == "scale":
        var total_elements = X.rows * d
        var sum_val: Scalar[dtype] = 0.0
        var ptr = X.data.unsafe_ptr()

        for i in range(total_elements):
            sum_val += ptr.unsafe_offset(i)[]

        var mean_val = sum_val / Scalar[dtype](total_elements)
        var sum_sq: Scalar[dtype] = 0.0
        for i in range(total_elements):
            var diff = ptr.unsafe_offset(i)[] - mean_val
            sum_sq += diff * diff

        var var_val = sum_sq / Scalar[dtype](total_elements)
        if var_val > Scalar[dtype](1e-12):
            return Scalar[dtype](1.0) / (Scalar[dtype](d) * var_val)
        return Scalar[dtype](1.0)

    raise InvalidParameterError.error(
        "gamma",
        "Expected 'scale', 'auto', or numeric value > 0; got '"
        + gamma_str
        + "'",
    )


def kernel_eval_rows[
    dtype: DType
](
    X1: Matrix[dtype],
    r1: Int,
    X2: Matrix[dtype],
    r2: Int,
    kernel: String,
    gamma: Scalar[dtype],
    degree: Int = 3,
    coef0: Scalar[dtype] = 0.0,
) raises -> Scalar[dtype]:
    """Evaluates kernel function between row r1 of X1 and row r2 of X2.

    Parameters:
        dtype: Numerical data type.

    Args:
        X1: First feature matrix.
        r1: Row index in X1.
        X2: Second feature matrix.
        r2: Row index in X2.
        kernel: Kernel routine ('linear', 'rbf', 'poly', 'sigmoid').
        gamma: Kernel scaling parameter.
        degree: Degree for polynomial kernel. Default 3.
        coef0: Independent term for poly and sigmoid kernels. Default 0.0.

    Returns:
        Scalar[dtype]: Kernel function value.

    Raises:
        InvalidParameterError: If kernel is unrecognized or degree < 1.
    """
    if kernel == "rbf":
        var d2 = _sq_dist_rows(X1, r1, X2, r2)
        return Scalar[dtype](exp(Float64(-gamma * d2)))
    elif kernel == "linear":
        return _dot_rows(X1, r1, X2, r2)
    elif kernel == "poly":
        if degree < 1:
            raise InvalidParameterError.error("degree", "Must be >= 1")
        var dot = _dot_rows(X1, r1, X2, r2)
        var base = gamma * dot + coef0
        var res = base
        for _ in range(degree - 1):
            res *= base
        return res
    elif kernel == "sigmoid":
        var dot = _dot_rows(X1, r1, X2, r2)
        return Scalar[dtype](tanh(Float64(gamma * dot + coef0)))
    else:
        raise InvalidParameterError.error(
            "kernel",
            "Unsupported kernel '"
            + kernel
            + "'. Expected 'linear', 'rbf', 'poly', or 'sigmoid'.",
        )


def kernel_eval_vec_row[
    dtype: DType
](
    v: List[Scalar[dtype]],
    X: Matrix[dtype],
    r: Int,
    kernel: String,
    gamma: Scalar[dtype],
    degree: Int = 3,
    coef0: Scalar[dtype] = 0.0,
) raises -> Scalar[dtype]:
    """Evaluates kernel function between vector v and row r of matrix X.

    Parameters:
        dtype: Numerical data type.

    Args:
        v: 1D feature vector of length d.
        X: Feature matrix with cols == d.
        r: Row index in X.
        kernel: Kernel routine ('linear', 'rbf', 'poly', 'sigmoid').
        gamma: Kernel scaling parameter.
        degree: Degree for polynomial kernel. Default 3.
        coef0: Independent term for poly and sigmoid kernels. Default 0.0.

    Returns:
        Scalar[dtype]: Kernel function value.
    """
    if kernel == "rbf":
        var d2 = _sq_dist_vec_row(v, X, r)
        return Scalar[dtype](exp(Float64(-gamma * d2)))
    elif kernel == "linear":
        return _dot_vec_row(v, X, r)
    elif kernel == "poly":
        if degree < 1:
            raise InvalidParameterError.error("degree", "Must be >= 1")
        var dot = _dot_vec_row(v, X, r)
        var base = gamma * dot + coef0
        var res = base
        for _ in range(degree - 1):
            res *= base
        return res
    elif kernel == "sigmoid":
        var dot = _dot_vec_row(v, X, r)
        return Scalar[dtype](tanh(Float64(gamma * dot + coef0)))
    else:
        raise InvalidParameterError.error(
            "kernel",
            "Unsupported kernel '"
            + kernel
            + "'. Expected 'linear', 'rbf', 'poly', or 'sigmoid'.",
        )


def compute_kernel_matrix[
    dtype: DType
](
    X1: Matrix[dtype],
    X2: Matrix[dtype],
    kernel: String,
    gamma: Scalar[dtype],
    degree: Int = 3,
    coef0: Scalar[dtype] = 0.0,
) raises -> Matrix[dtype]:
    """Computes pair-wise kernel matrix between X1 (N1 x D) and X2 (N2 x D).

    Parameters:
        dtype: Numerical data type.

    Args:
        X1: Matrix of shape (N1, D).
        X2: Matrix of shape (N2, D).
        kernel: Kernel routine ('linear', 'rbf', 'poly', 'sigmoid').
        gamma: Kernel scaling parameter.
        degree: Degree for polynomial kernel. Default 3.
        coef0: Independent term for poly and sigmoid kernels. Default 0.0.

    Returns:
        Matrix[dtype]: Matrix of shape (N1, N2).
    """
    if X1.cols != X2.cols:
        raise DimensionMismatchError.error(
            "X1.cols == X2.cols",
            "X1.cols == " + String(X1.cols) + ", X2.cols == " + String(X2.cols),
            "compute_kernel_matrix",
        )

    var n1 = X1.rows
    var n2 = X2.rows
    var K = Matrix[dtype](n1, n2, 0.0)

    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_eval_rows(
                X1, i, X2, j, kernel, gamma, degree, coef0
            )

    return K^


def compute_gram_matrix[
    dtype: DType
](
    X: Matrix[dtype],
    kernel: String,
    gamma: Scalar[dtype],
    degree: Int = 3,
    coef0: Scalar[dtype] = 0.0,
) raises -> Matrix[dtype]:
    """Computes symmetric Gram matrix K(X, X) with diagonal and symmetry shortcuts.

    Parameters:
        dtype: Numerical data type.

    Args:
        X: Matrix of shape (N, D).
        kernel: Kernel routine ('linear', 'rbf', 'poly', 'sigmoid').
        gamma: Kernel scaling parameter.
        degree: Degree for polynomial kernel. Default 3.
        coef0: Independent term for poly and sigmoid kernels. Default 0.0.

    Returns:
        Matrix[dtype]: Symmetric kernel matrix of shape (N, N).
    """
    var n = X.rows
    var K = Matrix[dtype](n, n, 0.0)

    if kernel == "rbf":
        for i in range(n):
            K[i, i] = Scalar[dtype](1.0)
            for j in range(i + 1, n):
                var val = Scalar[dtype](
                    exp(Float64(-gamma * _sq_dist_rows(X, i, X, j)))
                )
                K[i, j] = val
                K[j, i] = val
        return K^

    for i in range(n):
        K[i, i] = kernel_eval_rows(X, i, X, i, kernel, gamma, degree, coef0)
        for j in range(i + 1, n):
            var val = kernel_eval_rows(X, i, X, j, kernel, gamma, degree, coef0)
            K[i, j] = val
            K[j, i] = val

    return K^
