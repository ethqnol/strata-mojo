from std.math import abs, max, min
from ..core.matrix import Matrix
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct SMOResult[dtype: DType](Copyable, Movable):
    """Result container for kernel SVM dual solver."""

    var dual_coef: List[Scalar[Self.dtype]]
    var intercept: Scalar[Self.dtype]
    var support_indices: List[Int]
    var n_iter: Int
    var converged: Bool

    def __init__(
        out self,
        var dual_coef: List[Scalar[Self.dtype]],
        intercept: Scalar[Self.dtype],
        var support_indices: List[Int],
        n_iter: Int,
        converged: Bool,
    ):
        self.dual_coef = dual_coef^
        self.intercept = intercept
        self.support_indices = support_indices^
        self.n_iter = n_iter
        self.converged = converged

    def __init__(out self, *, copy: Self):
        self.dual_coef = copy.dual_coef.copy()
        self.intercept = copy.intercept
        self.support_indices = copy.support_indices.copy()
        self.n_iter = copy.n_iter
        self.converged = copy.converged

    def __init__(out self, *, deinit move: Self):
        self.dual_coef = move.dual_coef^
        self.intercept = move.intercept
        self.support_indices = move.support_indices^
        self.n_iter = move.n_iter
        self.converged = move.converged


def _smo_binary_svc[
    dtype: DType
](
    K: Matrix[dtype],
    y: List[Scalar[dtype]],
    C: Scalar[dtype],
    tol: Scalar[dtype] = 1e-3,
    max_iter: Int = 10000,
) raises -> SMOResult[dtype]:
    """Optimizes binary C-SVM dual problem via Sequential Minimal Optimization (SMO).

    Solves:
        min_alpha 1/2 alpha^T Q alpha - sum_i alpha_i
        s.t. 0 <= alpha_i <= C, sum_i y_i alpha_i = 0

    Args:
        K: Precomputed Gram matrix of shape (N, N).
        y: Binary targets in {-1.0, +1.0} of length N.
        C: Regularization parameter (C > 0).
        tol: Stopping tolerance on maximal violating pair duality gap. Default 1e-3.
        max_iter: Maximum number of coordinate pair updates. Default 10000.

    Returns:
        SMOResult containing dual coefficients (alpha * y), intercept, support indices.
    """
    var n = K.rows
    if n == 0 or K.cols != n:
        raise DimensionMismatchError.error(
            "K is square matrix",
            "K.rows == " + String(n) + ", K.cols == " + String(K.cols),
            "_smo_binary_svc",
        )
    if len(y) != n:
        raise DimensionMismatchError.error(
            "len(y) == K.rows",
            "len(y) == " + String(len(y)) + ", K.rows == " + String(n),
            "_smo_binary_svc",
        )
    if C <= 0:
        raise InvalidParameterError.error("C", "Must be positive")

    var alpha = List[Scalar[dtype]](capacity=n)
    var G = List[Scalar[dtype]](capacity=n)
    for _ in range(n):
        alpha.append(0.0)
        G.append(-1.0)

    var n_iter = 0
    var converged = False
    var eps: Scalar[dtype] = 1e-12

    while n_iter < max_iter:
        n_iter += 1

        # Find maximal violating pair (i, j)
        var i = -1
        var j = -1
        var f_max: Scalar[dtype] = -1e30
        var f_min: Scalar[dtype] = 1e30

        for t in range(n):
            var a_t = alpha[t]
            var y_t = y[t]
            var f_t = -y_t * G[t]

            # Up set: can increase alpha_t * y_t
            if (y_t > 0 and a_t < C - eps) or (y_t < 0 and a_t > eps):
                if f_t > f_max:
                    f_max = f_t
                    i = t

            # Low set: can decrease alpha_t * y_t
            if (y_t > 0 and a_t > eps) or (y_t < 0 and a_t < C - eps):
                if f_t < f_min:
                    f_min = f_t
                    j = t

        if i == -1 or j == -1 or (f_max - f_min <= tol):
            converged = True
            break

        var y_i = y[i]
        var y_j = y[j]
        var s = y_i * y_j

        # Compute box bounds for alpha_j
        var L: Scalar[dtype]
        var H: Scalar[dtype]
        if y_i != y_j:
            L = max(Scalar[dtype](0.0), alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(Scalar[dtype](0.0), alpha[i] + alpha[j] - C)
            H = min(C, alpha[i] + alpha[j])

        if L >= H:
            continue

        var k_ii = K[i, i]
        var k_jj = K[j, j]
        var k_ij = K[i, j]
        var eta = k_ii + k_jj - Scalar[dtype](2.0) * k_ij

        var alpha_j_new: Scalar[dtype]
        if eta > eps:
            var delta = (s * G[i] - G[j]) / eta
            alpha_j_new = alpha[j] + delta
            if alpha_j_new > H:
                alpha_j_new = H
            elif alpha_j_new < L:
                alpha_j_new = L
        else:
            # Degenerate curvature: evaluate dual objective at L and H
            var f_L = (G[j] - s * G[i]) * (L - alpha[j]) + Scalar[dtype](
                0.5
            ) * eta * (L - alpha[j]) * (L - alpha[j])
            var f_H = (G[j] - s * G[i]) * (H - alpha[j]) + Scalar[dtype](
                0.5
            ) * eta * (H - alpha[j]) * (H - alpha[j])
            if f_L < f_H - eps:
                alpha_j_new = L
            else:
                alpha_j_new = H

        var delta_j = alpha_j_new - alpha[j]
        if abs(delta_j) < eps:
            continue

        var alpha_i_new = alpha[i] - s * delta_j
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new

        # Update gradient cache: delta G_k = y_k * y_j * delta_j * (K_kj - K_ki)
        var c = y_j * delta_j
        var p_k_col_i = K.data.unsafe_ptr()
        var p_k_col_j = K.data.unsafe_ptr()

        for k in range(n):
            var k_ki = p_k_col_i.unsafe_offset(k * n + i)[]
            var k_kj = p_k_col_j.unsafe_offset(k * n + j)[]
            G[k] += y[k] * c * (k_kj - k_ki)

    # Compute bias intercept b
    var free_count = 0
    var b_sum: Scalar[dtype] = 0.0
    for t in range(n):
        if alpha[t] > eps and alpha[t] < C - eps:
            b_sum += -y[t] * G[t]
            free_count += 1

    var b: Scalar[dtype]
    if free_count > 0:
        b = b_sum / Scalar[dtype](free_count)
    else:
        var max_low: Scalar[dtype] = -1e30
        var min_up: Scalar[dtype] = 1e30
        for t in range(n):
            var f_t = -y[t] * G[t]
            if (y[t] > 0 and alpha[t] > eps) or (
                y[t] < 0 and alpha[t] < C - eps
            ):
                if f_t > max_low:
                    max_low = f_t
            if (y[t] > 0 and alpha[t] < C - eps) or (
                y[t] < 0 and alpha[t] > eps
            ):
                if f_t < min_up:
                    min_up = f_t
        if max_low > -1e29 and min_up < 1e29:
            b = (max_low + min_up) / Scalar[dtype](2.0)
        else:
            b = 0.0

    # Collect support vectors and dual coefficients (alpha * y)
    var support_indices = List[Int]()
    var dual_coef = List[Scalar[dtype]]()
    for t in range(n):
        if alpha[t] > eps:
            support_indices.append(t)
            dual_coef.append(alpha[t] * y[t])

    return SMOResult[dtype](dual_coef^, b, support_indices^, n_iter, converged)


def _kernel_svr_solver[
    dtype: DType
](
    K: Matrix[dtype],
    y: List[Scalar[dtype]],
    C: Scalar[dtype],
    epsilon: Scalar[dtype] = 0.1,
    fit_intercept: Bool = True,
    tol: Scalar[dtype] = 1e-3,
    max_iter: Int = 10000,
) raises -> SMOResult[dtype]:
    """Optimizes kernel Support Vector Regression (SVR) via dual coordinate descent.

    Solves:
        min_beta 1/2 beta^T (K + 1) beta - y^T beta + epsilon sum_i |beta_i|
        s.t. -C <= beta_i <= C

    Args:
        K: Gram matrix of shape (N, N).
        y: Target values of length N.
        C: Regularization penalty (C > 0).
        epsilon: Epsilon-insensitive loss tube width (>= 0). Default 0.1.
        fit_intercept: Whether to compute intercept via synthetic bias. Default True.
        tol: Stopping tolerance for relative gradient progress. Default 1e-3.
        max_iter: Maximum number of coordinate descent epochs. Default 10000.

    Returns:
        SMOResult containing dual coefficients (beta), intercept, support indices.
    """
    var n = K.rows
    if n == 0 or K.cols != n:
        raise DimensionMismatchError.error(
            "K is square matrix",
            "K.rows == " + String(n) + ", K.cols == " + String(K.cols),
            "_kernel_svr_solver",
        )
    if len(y) != n:
        raise DimensionMismatchError.error(
            "len(y) == K.rows",
            "len(y) == " + String(len(y)) + ", K.rows == " + String(n),
            "_kernel_svr_solver",
        )
    if C <= 0:
        raise InvalidParameterError.error("C", "Must be positive")
    if epsilon < 0:
        raise InvalidParameterError.error("epsilon", "Must be non-negative")

    var beta = List[Scalar[dtype]](capacity=n)
    var g = List[Scalar[dtype]](capacity=n)
    var bias_term = Scalar[dtype](1.0) if fit_intercept else Scalar[dtype](0.0)

    for i in range(n):
        beta.append(0.0)
        g.append(-y[i])

    var QD = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        QD.append(K[i, i] + bias_term)

    var n_iter = 0
    var converged = False
    var eps: Scalar[dtype] = 1e-12

    while n_iter < max_iter:
        n_iter += 1
        var max_delta: Scalar[dtype] = 0.0

        for i in range(n):
            var q_ii = QD[i]
            if q_ii < eps:
                continue

            var t = beta[i] - g[i] / q_ii
            var abs_t = abs(t)
            var z: Scalar[dtype] = 0.0

            # Soft-thresholding for epsilon-insensitive loss
            var threshold = epsilon / q_ii
            if abs_t > threshold:
                var sign_t: Scalar[dtype] = 1.0 if t > 0 else -1.0
                z = sign_t * (abs_t - threshold)

            if z > C:
                z = C
            elif z < -C:
                z = -C

            var delta = z - beta[i]
            if abs(delta) > max_delta:
                max_delta = abs(delta)

            if abs(delta) > eps:
                beta[i] = z
                # Update gradient: g_k += (K_ki + bias_term) * delta
                var p_col = K.data.unsafe_ptr()
                for k in range(n):
                    var q_ki = p_col.unsafe_offset(k * n + i)[] + bias_term
                    g[k] += q_ki * delta

        if max_delta <= tol:
            converged = True
            break

    # Calculate intercept and collect support vectors
    var intercept: Scalar[dtype] = 0.0
    if fit_intercept:
        for i in range(n):
            intercept += beta[i]

    var support_indices = List[Int]()
    var dual_coef = List[Scalar[dtype]]()
    for i in range(n):
        if abs(beta[i]) > eps:
            support_indices.append(i)
            dual_coef.append(beta[i])

    return SMOResult[dtype](
        dual_coef^, intercept, support_indices^, n_iter, converged
    )
