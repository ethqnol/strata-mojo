from std.math import abs, max, min
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
            "len(y) == X.rows",
            "len(y) == " + String(len(y)) + ", X.rows == " + String(l),
            "_dual_coordinate_descent_svm",
        )

    var is_l2_loss = loss == "squared_hinge"
    var D_ii: Scalar[dtype] = Scalar[dtype](1.0) / (
        Scalar[dtype](2.0) * C
    ) if is_l2_loss else Scalar[dtype](0.0)
    var U: Scalar[dtype] = Scalar[dtype](1e300) if is_l2_loss else C

    # Precalculate Q_ii + D_ii for each sample i
    var QD = List[Scalar[dtype]](capacity=l)
    var bias_sq = (
        intercept_scaling
        * intercept_scaling if fit_intercept else Scalar[dtype](0.0)
    )

    for i in range(l):
        var q_ii: Scalar[dtype] = 0.0
        for j in range(d):
            var val = X[i, j]
            q_ii += val * val
        q_ii += bias_sq
        QD.append(q_ii + D_ii)

    # Dual variables alpha and primal weight vector w
    var alpha = List[Scalar[dtype]](capacity=l)
    for _ in range(l):
        alpha.append(0.0)

    var w = List[Scalar[dtype]](capacity=d)
    for _ in range(d):
        w.append(0.0)

    var b: Scalar[dtype] = 0.0
    var n_iter = 0

    for epoch in range(max_iter):
        n_iter = epoch + 1
        var perm = permutation(l, random_state + epoch * 37)

        var pg_max: Scalar[dtype] = -Scalar[dtype](1e300)
        var pg_min: Scalar[dtype] = Scalar[dtype](1e300)

        for s in range(l):
            var i = perm[s]
            var yi = y[i]
            var qd_i = QD[i]

            # Compute dot product: w^T x_i + b * intercept_scaling
            var dot: Scalar[dtype] = 0.0
            for j in range(d):
                dot += w[j] * X[i, j]
            if fit_intercept:
                dot += b * intercept_scaling

            # Gradient: G = y_i * (w^T x_i) - 1 + D_ii * alpha_i
            var G = yi * dot - Scalar[dtype](1.0) + D_ii * alpha[i]

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
