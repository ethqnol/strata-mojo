from std.math import abs, max, min, inf
from ..core.matrix import Matrix
from ..utils.random import PRNG, permutation
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


def _dual_coordinate_descent_svm[
    dtype: DType = DType.float64
](
    X: Matrix[dtype],
    y: List[Scalar[dtype]],
    C: Scalar[dtype],
    loss: String = "squared_hinge",
    fit_intercept: Bool = True,
    intercept_scaling: Scalar[dtype] = 1.0,
    max_iter: Int = 1000,
    tol: Scalar[dtype] = 1e-4,
    random_state: Int = 42,
) raises -> Tuple[List[Scalar[dtype]], Scalar[dtype], Int]:
    """Optimizes the L2-regularized linear SVM objective via Dual Coordinate Descent (LIBLINEAR).

    Solves the dual quadratic program:
    $$
    \\min_{\\alpha} \\frac{1}{2} \\alpha^T \\bar{Q} \\alpha - \\sum_{i=1}^l \\alpha_i
    \\quad \\text{s.t.} \\quad 0 \\le \\alpha_i \\le U
    $$
    where:
    - For L1 loss ('hinge'): U = C, D_ii = 0
    - For L2 loss ('squared_hinge'): U = infinity, D_ii = 1 / (2C)
    - Primal weight vector: w = sum_i alpha_i * y_i * x_i

    Parameters:
        dtype: Computational data type. Default DType.float64.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target labels encoded as -1.0 and +1.0 of length n_samples.
        C: Regularization penalty parameter (C > 0).
        loss: Loss function ('hinge' or 'squared_hinge'). Default 'squared_hinge'.
        fit_intercept: Whether to compute bias intercept. Default True.
        intercept_scaling: Scaling multiplier for synthetic bias dimension. Default 1.0.
        max_iter: Maximum coordinate descent epochs. Default 1000.
        tol: Stopping tolerance for projected gradient difference. Default 1e-4.
        random_state: Seed for random sample permutation. Default 42.

    Returns:
        Tuple of (weights vector of length d, intercept scalar, iterations run).
    """
    var l = X.rows
    var d = X.cols

    if l == 0 or d == 0:
        raise DimensionMismatchError.error(
            "X.rows > 0 and X.cols > 0",
            "X.rows == " + String(l) + ", X.cols == " + String(d),
            "_dual_coordinate_descent_svm",
        )

    if len(y) != l:
        raise DimensionMismatchError.error(
            "len(y) == " + String(l),
            "len(y) == " + String(len(y)),
            "_dual_coordinate_descent_svm",
        )

    if C <= 0:
        raise InvalidParameterError.error(
            "C", "C must be positive, got " + String(C)
        )

    if tol <= 0:
        raise InvalidParameterError.error(
            "tol", "tol must be positive, got " + String(tol)
        )

    if max_iter <= 0:
        raise InvalidParameterError.error(
            "max_iter", "max_iter must be positive, got " + String(max_iter)
        )

    var diag_D: Scalar[dtype]
    var U: Scalar[dtype]

    if loss == "squared_hinge" or loss == "l2":
        diag_D = Scalar[dtype](0.5) / C
        U = inf[dtype]()
    elif loss == "hinge" or loss == "l1":
        diag_D = Scalar[dtype](0.0)
        U = C
    else:
        raise InvalidParameterError.error(
            "loss",
            "Unknown loss '" + loss + "'. Expected 'squared_hinge' or 'hinge'.",
        )

    # Precompute diagonal of Q: Q_ii = ||x_i||^2 + intercept_scaling^2
    var QD = List[Scalar[dtype]](capacity=l)
    for i in range(l):
        var qd_val: Scalar[dtype] = 0.0
        for j in range(d):
            var val = X[i, j]
            qd_val += val * val
        if fit_intercept:
            qd_val += intercept_scaling * intercept_scaling
        QD.append(qd_val + diag_D)

    var alpha = List[Scalar[dtype]](capacity=l)
    for _ in range(l):
        alpha.append(Scalar[dtype](0.0))

    var w = List[Scalar[dtype]](capacity=d)
    for _ in range(d):
        w.append(Scalar[dtype](0.0))

    var b: Scalar[dtype] = 0.0
    var n_iter = 0

    var index = List[Int](capacity=l)
    for i in range(l):
        index.append(i)

    for iter in range(max_iter):
        n_iter = iter + 1
        var pg_max = -inf[dtype]()
        var pg_min = inf[dtype]()

        # Shuffle sample traversal order each epoch
        var perm = permutation(l, random_state + iter * 37)
        var permuted_index = List[Int](capacity=l)
        for k in range(l):
            permuted_index.append(index[perm[k]])
        for k in range(l):
            index[k] = permuted_index[k]

        for s in range(l):
            var i = index[s]
            var yi = y[i]
            var qd_i = QD[i]

            # Compute margin: w^T x_i + bias
            var wx_i: Scalar[dtype] = 0.0
            for j in range(d):
                wx_i += w[j] * X[i, j]
            if fit_intercept:
                wx_i += b * intercept_scaling

            # Gradient: G = y_i * (w^T x_i + bias) - 1 + D_ii * alpha_i
            var G = yi * wx_i - Scalar[dtype](1.0) + diag_D * alpha[i]

            # Projected gradient
            var PG: Scalar[dtype] = 0.0
            if alpha[i] == 0.0:
                if G < 0.0:
                    PG = G
            elif alpha[i] == U:
                if G > 0.0:
                    PG = G
            else:
                PG = G

            if PG > pg_max:
                pg_max = PG
            if PG < pg_min:
                pg_min = PG

            # Coordinate descent step if projected gradient is non-zero
            if abs(PG) > Scalar[dtype](1e-12):
                var alpha_old = alpha[i]
                var alpha_new = alpha_old - G / qd_i

                if alpha_new < Scalar[dtype](0.0):
                    alpha_new = Scalar[dtype](0.0)
                elif alpha_new > U:
                    alpha_new = U

                var d_alpha = alpha_new - alpha_old
                alpha[i] = alpha_new

                if d_alpha != Scalar[dtype](0.0):
                    var scale = d_alpha * yi
                    for j in range(d):
                        w[j] += scale * X[i, j]
                    if fit_intercept:
                        b += scale * intercept_scaling

        # Check stopping criterion: (pg_max - pg_min) <= tol
        if (pg_max - pg_min) <= tol:
            break

    var final_intercept = b * intercept_scaling if fit_intercept else Scalar[
        dtype
    ](0.0)
    return (w^, final_intercept, n_iter)


def _dual_coordinate_descent_svr[
    dtype: DType = DType.float64
](
    X: Matrix[dtype],
    y: List[Scalar[dtype]],
    C: Scalar[dtype],
    p: Scalar[dtype] = 0.0,
    loss: String = "epsilon_insensitive",
    fit_intercept: Bool = True,
    intercept_scaling: Scalar[dtype] = 1.0,
    max_iter: Int = 1000,
    tol: Scalar[dtype] = 1e-4,
    random_state: Int = 42,
) raises -> Tuple[List[Scalar[dtype]], Scalar[dtype], Int]:
    """Optimizes the L2-regularized linear Support Vector Regression via Dual Coordinate Descent (LIBLINEAR).

    Solves the dual quadratic program for epsilon-insensitive regression (Ho & Lin, 2012):
    $$
    \\min_{\\beta} \\frac{1}{2} \\beta^T (Q + \\text{diag}(\\lambda)) \\beta - y^T \\beta + \\epsilon \\sum_{i=1}^l |\\beta_i|
    \\quad \\text{s.t.} \\quad -U \\le \\beta_i \\le U
    $$
    where:
    - For L1 loss ('epsilon_insensitive'): U = C, lambda = 0
    - For L2 loss ('squared_epsilon_insensitive'): U = infinity, lambda = 1 / (2C)
    - Primal weights: w = sum_i beta_i * x_i

    Parameters:
        dtype: Computational data type. Default DType.float64.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Continuous target vector of length n_samples.
        C: Regularization penalty parameter (C > 0).
        p: Epsilon tube width (epsilon >= 0). Default 0.0.
        loss: Loss variant ('epsilon_insensitive' or 'squared_epsilon_insensitive'). Default 'epsilon_insensitive'.
        fit_intercept: Whether to compute bias intercept. Default True.
        intercept_scaling: Scaling multiplier for synthetic bias dimension. Default 1.0.
        max_iter: Maximum coordinate descent epochs. Default 1000.
        tol: Stopping tolerance based on relative gradient norm reduction. Default 1e-4.
        random_state: Seed for random sample permutation. Default 42.

    Returns:
        Tuple of (weights vector of length d, intercept scalar, iterations run).
    """
    var l = X.rows
    var d = X.cols

    if l == 0 or d == 0:
        raise DimensionMismatchError.error(
            "X.rows > 0 and X.cols > 0",
            "X.rows == " + String(l) + ", X.cols == " + String(d),
            "_dual_coordinate_descent_svr",
        )

    if len(y) != l:
        raise DimensionMismatchError.error(
            "len(y) == " + String(l),
            "len(y) == " + String(len(y)),
            "_dual_coordinate_descent_svr",
        )

    if C <= 0:
        raise InvalidParameterError.error(
            "C", "C must be positive, got " + String(C)
        )

    if p < 0:
        raise InvalidParameterError.error(
            "epsilon", "epsilon must be non-negative, got " + String(p)
        )

    if tol <= 0:
        raise InvalidParameterError.error(
            "tol", "tol must be positive, got " + String(tol)
        )

    if max_iter <= 0:
        raise InvalidParameterError.error(
            "max_iter", "max_iter must be positive, got " + String(max_iter)
        )

    var lambda_val: Scalar[dtype]
    var upper_bound: Scalar[dtype]

    if loss == "epsilon_insensitive" or loss == "l1":
        lambda_val = Scalar[dtype](0.0)
        upper_bound = C
    elif loss == "squared_epsilon_insensitive" or loss == "l2":
        lambda_val = Scalar[dtype](0.5) / C
        upper_bound = inf[dtype]()
    else:
        raise InvalidParameterError.error(
            "loss",
            "Unknown loss '"
            + loss
            + "'. Expected 'epsilon_insensitive' or"
            " 'squared_epsilon_insensitive'.",
        )

    # Precompute Q_ii = ||x_i||^2 + intercept_scaling^2 and H_i = Q_ii + lambda
    var H = List[Scalar[dtype]](capacity=l)
    for i in range(l):
        var qd_val: Scalar[dtype] = 0.0
        for j in range(d):
            var val = X[i, j]
            qd_val += val * val
        if fit_intercept:
            qd_val += intercept_scaling * intercept_scaling
        H.append(qd_val + lambda_val)

    var beta = List[Scalar[dtype]](capacity=l)
    for _ in range(l):
        beta.append(Scalar[dtype](0.0))

    var w = List[Scalar[dtype]](capacity=d)
    for _ in range(d):
        w.append(Scalar[dtype](0.0))

    var b: Scalar[dtype] = 0.0
    var n_iter = 0
    var active_size = l

    var index = List[Int](capacity=l)
    for i in range(l):
        index.append(i)

    var Gmax_old = inf[dtype]()
    var Gnorm1_init: Scalar[dtype] = -1.0

    for iter in range(max_iter):
        n_iter = iter + 1
        var Gmax_new: Scalar[dtype] = 0.0
        var Gnorm1_new: Scalar[dtype] = 0.0

        # Permute active indices using deterministic random shuffling
        var perm = permutation(active_size, random_state + iter * 37)
        var permuted_index = List[Int](capacity=active_size)
        for k in range(active_size):
            permuted_index.append(index[perm[k]])
        for k in range(active_size):
            index[k] = permuted_index[k]

        var s = 0
        while s < active_size:
            var i = index[s]

            # Compute prediction: w^T x_i + bias
            var wx_i: Scalar[dtype] = 0.0
            for j in range(d):
                wx_i += w[j] * X[i, j]
            if fit_intercept:
                wx_i += b * intercept_scaling

            var G = -y[i] + lambda_val * beta[i] + wx_i
            var H_i = H[i]

            var Gp = G + p
            var Gn = G - p
            var violation: Scalar[dtype] = 0.0

            if beta[i] == Scalar[dtype](0.0):
                if Gp < Scalar[dtype](0.0):
                    violation = -Gp
                elif Gn > Scalar[dtype](0.0):
                    violation = Gn
                elif Gp > Gmax_old and Gn < -Gmax_old:
                    active_size -= 1
                    var tmp = index[s]
                    index[s] = index[active_size]
                    index[active_size] = tmp
                    continue
            elif beta[i] >= upper_bound:
                if Gp > Scalar[dtype](0.0):
                    violation = Gp
                elif Gp < -Gmax_old:
                    active_size -= 1
                    var tmp = index[s]
                    index[s] = index[active_size]
                    index[active_size] = tmp
                    continue
            elif beta[i] <= -upper_bound:
                if Gn < Scalar[dtype](0.0):
                    violation = -Gn
                elif Gn > Gmax_old:
                    active_size -= 1
                    var tmp = index[s]
                    index[s] = index[active_size]
                    index[active_size] = tmp
                    continue
            elif beta[i] > Scalar[dtype](0.0):
                violation = abs(Gp)
            else:
                violation = abs(Gn)

            if violation > Gmax_new:
                Gmax_new = violation
            Gnorm1_new += violation

            # Compute Newton step direction
            var step: Scalar[dtype]
            if Gp < H_i * beta[i]:
                step = -Gp / H_i
            elif Gn > H_i * beta[i]:
                step = -Gn / H_i
            else:
                step = -beta[i]

            if abs(step) >= Scalar[dtype](1e-12):
                var beta_old = beta[i]
                var beta_cand = beta_old + step
                if beta_cand > upper_bound:
                    beta_cand = upper_bound
                elif beta_cand < -upper_bound:
                    beta_cand = -upper_bound
                beta[i] = beta_cand
                var delta = beta[i] - beta_old

                if delta != Scalar[dtype](0.0):
                    for j in range(d):
                        w[j] += delta * X[i, j]
                    if fit_intercept:
                        b += delta * intercept_scaling

            s += 1

        if iter == 0:
            Gnorm1_init = Gnorm1_new

        if Gnorm1_init == Scalar[dtype](0.0) or (
            Gnorm1_init > Scalar[dtype](0.0) and Gnorm1_new <= tol * Gnorm1_init
        ):
            if active_size == l:
                break
            else:
                active_size = l
                Gmax_old = inf[dtype]()
                continue

        Gmax_old = Gmax_new

    var final_intercept = b * intercept_scaling if fit_intercept else Scalar[
        dtype
    ](0.0)
    return (w^, final_intercept, n_iter)
