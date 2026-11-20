from std.math import sqrt, log, exp, cos, sin, isnan, isinf
from ..core.matrix import Matrix
from ..utils.random import PRNG
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError

comptime LN_2PI: Float64 = 1.8378770664093454835606594728112
comptime TWO_PI: Float64 = 6.283185307179586476925286766559
comptime EPS_NK: Float64 = 10.0 * 2.220446049250313e-16


struct GaussianStats[dtype: DType = DType.float64](Copyable, Movable):
    """Effective component weights and weighted means from M-step aggregation.

    Parameters:
        dtype: Scalar floating-point data type. Default DType.float64.

    Attributes:
        nk: Effective sample counts per component of length $K$.
        means: Weighted component centers of shape $(K, D)$.
    """

    var nk: List[Scalar[Self.dtype]]
    var means: Matrix[Self.dtype]

    def __init__(
        out self,
        var nk: List[Scalar[Self.dtype]],
        var means: Matrix[Self.dtype],
    ):
        """Initialize a GaussianStats container."""
        self.nk = nk^
        self.means = means^

    def __init__(out self, *, copy: Self):
        """Copy an existing GaussianStats container."""
        self.nk = copy.nk.copy()
        self.means = copy.means.copy()

    def __init__(out self, *, deinit move: Self):
        """Move an existing GaussianStats container."""
        self.nk = move.nk^
        self.means = move.means^


struct EStepResult[dtype: DType = DType.float64](Copyable, Movable):
    """Expectation-step outputs: marginal log-densities, log-responsibilities, and mean bound.

    Parameters:
        dtype: Scalar floating-point data type. Default DType.float64.

    Attributes:
        log_prob_norm: Log probability density $\\ln p(x_i)$ for each sample of length $N$.
        log_resp: Log posterior responsibilities $\\ln r_{ik}$ of shape $(N, K)$.
        mean_log_prob: Mean log-likelihood lower bound across samples.
    """

    var log_prob_norm: List[Scalar[Self.dtype]]
    var log_resp: Matrix[Self.dtype]
    var mean_log_prob: Float64

    def __init__(
        out self,
        var log_prob_norm: List[Scalar[Self.dtype]],
        var log_resp: Matrix[Self.dtype],
        mean_log_prob: Float64,
    ):
        """Initialize an EStepResult container."""
        self.log_prob_norm = log_prob_norm^
        self.log_resp = log_resp^
        self.mean_log_prob = mean_log_prob

    def __init__(out self, *, copy: Self):
        """Copy an existing EStepResult container."""
        self.log_prob_norm = copy.log_prob_norm.copy()
        self.log_resp = copy.log_resp.copy()
        self.mean_log_prob = copy.mean_log_prob

    def __init__(out self, *, deinit move: Self):
        """Move an existing EStepResult container."""
        self.log_prob_norm = move.log_prob_norm^
        self.log_resp = move.log_resp^
        self.mean_log_prob = move.mean_log_prob


def _copy_matrix_list[
    dtype: DType
](src: List[Matrix[dtype]]) -> List[Matrix[dtype]]:
    """Deep-copy a list of dense Matrix containers."""
    var out = List[Matrix[dtype]](capacity=len(src))
    for i in range(len(src)):
        out.append(src[i].copy())
    return out^


struct _GMMState[dtype: DType = DType.float64](Copyable, Movable):
    """Internal snapshot of fitted mixture parameters across EM restarts."""

    var weights: List[Scalar[Self.dtype]]
    var means: Matrix[Self.dtype]
    var covariances: List[Matrix[Self.dtype]]
    var precisions_cholesky: List[Matrix[Self.dtype]]

    def __init__(
        out self,
        var weights: List[Scalar[Self.dtype]],
        var means: Matrix[Self.dtype],
        var covariances: List[Matrix[Self.dtype]],
        var precisions_cholesky: List[Matrix[Self.dtype]],
    ):
        self.weights = weights^
        self.means = means^
        self.covariances = covariances^
        self.precisions_cholesky = precisions_cholesky^

    def __init__(out self, *, copy: Self):
        self.weights = copy.weights.copy()
        self.means = copy.means.copy()
        self.covariances = _copy_matrix_list(copy.covariances)
        self.precisions_cholesky = _copy_matrix_list(copy.precisions_cholesky)

    def __init__(out self, *, deinit move: Self):
        self.weights = move.weights^
        self.means = move.means^
        self.covariances = move.covariances^
        self.precisions_cholesky = move.precisions_cholesky^


def _check_positive_finite(val: Float64, name: String, context: String) raises:
    """Validate that a scalar value is finite and strictly positive."""
    if isnan(val) or isinf(val) or val <= 0.0:
        raise InvalidParameterError.error(
            name,
            context
            + " requires finite strictly positive values, got "
            + String(val),
        )


def _estimate_gaussian_nk_and_means[
    dtype: DType
](X: Matrix[dtype], resp: Matrix[dtype]) raises -> GaussianStats[dtype]:
    """Compute effective sample counts and weighted component means."""
    if X.rows == 0 or X.cols == 0:
        raise DimensionMismatchError.error(
            "non-empty X",
            "shape (" + String(X.rows) + ", " + String(X.cols) + ")",
            "_estimate_gaussian_nk_and_means",
        )
    if resp.rows != X.rows or resp.cols == 0:
        raise DimensionMismatchError.error(
            "resp shape (" + String(X.rows) + ", K > 0)",
            "shape (" + String(resp.rows) + ", " + String(resp.cols) + ")",
            "_estimate_gaussian_nk_and_means",
        )

    var N = X.rows
    var D = X.cols
    var K = resp.cols

    var nk_f64 = List[Float64](capacity=K)
    for _ in range(K):
        nk_f64.append(0.0)
    var sum_rx = Matrix[DType.float64](K, D, 0.0)

    for i in range(N):
        for k in range(K):
            var r = Float64(resp[i, k])
            if isnan(r) or isinf(r) or r < 0.0:
                raise InvalidParameterError.error(
                    "resp",
                    "Responsibilities must be finite and non-negative",
                )
            nk_f64[k] += r
            if r != 0.0:
                for j in range(D):
                    sum_rx[k, j] += r * Float64(X[i, j])

    var nk = List[Scalar[dtype]](capacity=K)
    var means = Matrix[dtype](K, D, 0)
    for k in range(K):
        var nk_reg = nk_f64[k] + EPS_NK
        nk.append(Scalar[dtype](nk_reg))
        var inv_nk = 1.0 / nk_reg
        for j in range(D):
            means[k, j] = Scalar[dtype](sum_rx[k, j] * inv_nk)

    return GaussianStats[dtype](nk^, means^)


def _validate_cov_inputs[
    dtype: DType
](
    X: Matrix[dtype],
    resp: Matrix[dtype],
    nk: List[Scalar[dtype]],
    means: Matrix[dtype],
    reg_covar: Float64,
    context: String,
) raises:
    """Validate shapes and regularization for covariance estimation routines."""
    if X.rows == 0 or X.cols == 0:
        raise DimensionMismatchError.error(
            "non-empty X",
            "shape (" + String(X.rows) + ", " + String(X.cols) + ")",
            context,
        )
    if resp.rows != X.rows or resp.cols == 0:
        raise DimensionMismatchError.error(
            "resp shape (" + String(X.rows) + ", K > 0)",
            "shape (" + String(resp.rows) + ", " + String(resp.cols) + ")",
            context,
        )
    var K = resp.cols
    var D = X.cols
    if len(nk) != K:
        raise DimensionMismatchError.error(
            "len(nk) == " + String(K),
            "len(nk) == " + String(len(nk)),
            context,
        )
    if means.rows != K or means.cols != D:
        raise DimensionMismatchError.error(
            "means shape (" + String(K) + ", " + String(D) + ")",
            "shape (" + String(means.rows) + ", " + String(means.cols) + ")",
            context,
        )
    if isnan(reg_covar) or isinf(reg_covar) or reg_covar < 0.0:
        raise InvalidParameterError.error(
            "reg_covar",
            "reg_covar must be finite and non-negative, got "
            + String(reg_covar),
        )
    for k in range(K):
        _check_positive_finite(Float64(nk[k]), "nk", context)


def _estimate_covariances_full[
    dtype: DType
](
    X: Matrix[dtype],
    resp: Matrix[dtype],
    nk: List[Scalar[dtype]],
    means: Matrix[dtype],
    reg_covar: Float64,
) raises -> List[Matrix[dtype]]:
    """Estimate regularized full covariance matrices for each component."""
    _validate_cov_inputs(
        X, resp, nk, means, reg_covar, "_estimate_covariances_full"
    )
    var N = X.rows
    var D = X.cols
    var K = resp.cols

    var covs = List[Matrix[dtype]](capacity=K)
    var diff = List[Float64](capacity=D)
    for _ in range(D):
        diff.append(0.0)

    for k in range(K):
        var scatter = Matrix[DType.float64](D, D, 0.0)
        for i in range(N):
            var w = Float64(resp[i, k])
            if w == 0.0:
                continue
            for j in range(D):
                diff[j] = Float64(X[i, j]) - Float64(means[k, j])
            for r in range(D):
                var wdr = w * diff[r]
                for c in range(r, D):
                    scatter[r, c] += wdr * diff[c]

        var inv_nk = 1.0 / Float64(nk[k])
        var cov_mat = Matrix[dtype](D, D, 0)
        for r in range(D):
            cov_mat[r, r] = Scalar[dtype](scatter[r, r] * inv_nk + reg_covar)
            for c in range(r + 1, D):
                var val = Scalar[dtype](scatter[r, c] * inv_nk)
                cov_mat[r, c] = val
                cov_mat[c, r] = val
        covs.append(cov_mat^)

    return covs^


def _estimate_covariances_tied[
    dtype: DType
](
    X: Matrix[dtype],
    resp: Matrix[dtype],
    nk: List[Scalar[dtype]],
    means: Matrix[dtype],
    reg_covar: Float64,
) raises -> Matrix[dtype]:
    """Estimate a single pooled regularized covariance matrix shared across components.
    """
    _validate_cov_inputs(
        X, resp, nk, means, reg_covar, "_estimate_covariances_tied"
    )
    var N = X.rows
    var D = X.cols
    var K = resp.cols

    var total_nk: Float64 = 0.0
    for k in range(K):
        total_nk += Float64(nk[k])

    var scatter = Matrix[DType.float64](D, D, 0.0)
    var diff = List[Float64](capacity=D)
    for _ in range(D):
        diff.append(0.0)

    for k in range(K):
        for i in range(N):
            var w = Float64(resp[i, k])
            if w == 0.0:
                continue
            for j in range(D):
                diff[j] = Float64(X[i, j]) - Float64(means[k, j])
            for r in range(D):
                var wdr = w * diff[r]
                for c in range(r, D):
                    scatter[r, c] += wdr * diff[c]

    var inv_total = 1.0 / total_nk
    var cov_mat = Matrix[dtype](D, D, 0)
    for r in range(D):
        cov_mat[r, r] = Scalar[dtype](scatter[r, r] * inv_total + reg_covar)
        for c in range(r + 1, D):
            var val = Scalar[dtype](scatter[r, c] * inv_total)
            cov_mat[r, c] = val
            cov_mat[c, r] = val

    return cov_mat^


def _estimate_covariances_diag[
    dtype: DType
](
    X: Matrix[dtype],
    resp: Matrix[dtype],
    nk: List[Scalar[dtype]],
    means: Matrix[dtype],
    reg_covar: Float64,
) raises -> Matrix[dtype]:
    """Estimate regularized diagonal covariance vectors of shape (K, D)."""
    _validate_cov_inputs(
        X, resp, nk, means, reg_covar, "_estimate_covariances_diag"
    )
    var N = X.rows
    var D = X.cols
    var K = resp.cols

    var cov_diag = Matrix[dtype](K, D, 0)
    for k in range(K):
        var accum = List[Float64](capacity=D)
        for _ in range(D):
            accum.append(0.0)
        for i in range(N):
            var w = Float64(resp[i, k])
            if w == 0.0:
                continue
            for j in range(D):
                var d = Float64(X[i, j]) - Float64(means[k, j])
                accum[j] += w * d * d
        var inv_nk = 1.0 / Float64(nk[k])
        for j in range(D):
            cov_diag[k, j] = Scalar[dtype](accum[j] * inv_nk + reg_covar)

    return cov_diag^


def _estimate_covariances_spherical[
    dtype: DType
](
    X: Matrix[dtype],
    resp: Matrix[dtype],
    nk: List[Scalar[dtype]],
    means: Matrix[dtype],
    reg_covar: Float64,
) raises -> List[Scalar[dtype]]:
    """Estimate regularized isotropic scalar variances of length K."""
    _validate_cov_inputs(
        X, resp, nk, means, reg_covar, "_estimate_covariances_spherical"
    )
    var N = X.rows
    var D = X.cols
    var K = resp.cols

    var cov_sph = List[Scalar[dtype]](capacity=K)
    for k in range(K):
        var total_sq: Float64 = 0.0
        for i in range(N):
            var w = Float64(resp[i, k])
            if w == 0.0:
                continue
            var row_sq: Float64 = 0.0
            for j in range(D):
                var d = Float64(X[i, j]) - Float64(means[k, j])
                row_sq += d * d
            total_sq += w * row_sq
        var var_k = total_sq / (Float64(D) * Float64(nk[k])) + reg_covar
        cov_sph.append(Scalar[dtype](var_k))

    return cov_sph^


def _cholesky_lower[dtype: DType](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """Compute the lower-triangular Cholesky factor L such that A = L L^T."""
    if A.rows == 0 or A.rows != A.cols:
        raise DimensionMismatchError.error(
            "non-empty square matrix (rows == cols)",
            "shape (" + String(A.rows) + ", " + String(A.cols) + ")",
            "_cholesky_lower",
        )

    var D = A.rows
    var L_f64 = Matrix[DType.float64](D, D, 0.0)

    for r in range(D):
        for c in range(r + 1):
            var s = Float64(A[r, c])
            if isnan(s) or isinf(s):
                raise InvalidParameterError.error(
                    "A", "Covariance matrix contains NaN or infinite values"
                )
            for m in range(c):
                s -= L_f64[r, m] * L_f64[c, m]
            if r == c:
                if s <= 0.0 or isnan(s) or isinf(s):
                    raise InvalidParameterError.error(
                        "A",
                        "Matrix is not positive-definite (non-positive pivot"
                        " at index "
                        + String(r)
                        + ")",
                    )
                L_f64[r, r] = sqrt(s)
            else:
                L_f64[r, c] = s / L_f64[c, c]

    return L_f64.cast[dtype]()


def _precision_cholesky_from_lower[
    dtype: DType
](L: Matrix[dtype]) raises -> Matrix[dtype]:
    """Invert lower-triangular Cholesky factor L and return upper-triangular P = (L^-1)^T.
    """
    if L.rows == 0 or L.rows != L.cols:
        raise DimensionMismatchError.error(
            "non-empty square matrix (rows == cols)",
            "shape (" + String(L.rows) + ", " + String(L.cols) + ")",
            "_precision_cholesky_from_lower",
        )

    var D = L.rows
    var P_f64 = Matrix[DType.float64](D, D, 0.0)

    for r in range(D):
        _check_positive_finite(
            Float64(L[r, r]), "L", "_precision_cholesky_from_lower"
        )

    for c in range(D):
        P_f64[c, c] = 1.0 / Float64(L[c, c])
        for r in range(c + 1, D):
            var s: Float64 = 0.0
            for m in range(c, r):
                s -= Float64(L[r, m]) * P_f64[c, m]
            P_f64[c, r] = s / Float64(L[r, r])

    return P_f64.cast[dtype]()


def _compute_precision_cholesky_full[
    dtype: DType
](covariances: List[Matrix[dtype]]) raises -> List[Matrix[dtype]]:
    """Compute upper-triangular precision Cholesky factors for full covariances.
    """
    var K = len(covariances)
    if K == 0:
        raise DimensionMismatchError.error(
            "non-empty covariances list",
            "len == 0",
            "_compute_precision_cholesky_full",
        )
    var prec_chols = List[Matrix[dtype]](capacity=K)
    for k in range(K):
        var L = _cholesky_lower(covariances[k])
        prec_chols.append(_precision_cholesky_from_lower(L))
    return prec_chols^


def _compute_precision_cholesky_tied[
    dtype: DType
](covariance: Matrix[dtype]) raises -> Matrix[dtype]:
    """Compute upper-triangular precision Cholesky factor for tied covariance.
    """
    var L = _cholesky_lower(covariance)
    return _precision_cholesky_from_lower(L)


def _compute_precision_cholesky_diag[
    dtype: DType
](covariances: Matrix[dtype]) raises -> Matrix[dtype]:
    """Compute reciprocal standard deviations 1 / sqrt(cov) for diagonal covariances.
    """
    if covariances.rows == 0 or covariances.cols == 0:
        raise DimensionMismatchError.error(
            "non-empty covariances matrix",
            "shape ("
            + String(covariances.rows)
            + ", "
            + String(covariances.cols)
            + ")",
            "_compute_precision_cholesky_diag",
        )
    var K = covariances.rows
    var D = covariances.cols
    var prec_chol = Matrix[dtype](K, D, 0)
    for k in range(K):
        for j in range(D):
            var v = Float64(covariances[k, j])
            _check_positive_finite(
                v, "covariances", "_compute_precision_cholesky_diag"
            )
            prec_chol[k, j] = Scalar[dtype](1.0 / sqrt(v))
    return prec_chol^


def _compute_precision_cholesky_spherical[
    dtype: DType
](covariances: List[Scalar[dtype]]) raises -> List[Scalar[dtype]]:
    """Compute reciprocal standard deviations 1 / sqrt(cov) for spherical covariances.
    """
    var K = len(covariances)
    if K == 0:
        raise DimensionMismatchError.error(
            "non-empty spherical covariances",
            "len == 0",
            "_compute_precision_cholesky_spherical",
        )
    var prec_chol = List[Scalar[dtype]](capacity=K)
    for k in range(K):
        var v = Float64(covariances[k])
        _check_positive_finite(
            v, "covariances", "_compute_precision_cholesky_spherical"
        )
        prec_chol.append(Scalar[dtype](1.0 / sqrt(v)))
    return prec_chol^


def _compute_precisions_from_cholesky_tied[
    dtype: DType
](prec_chol: Matrix[dtype]) raises -> Matrix[dtype]:
    """Recover full precision matrix P P^T from upper-triangular factor P."""
    if prec_chol.rows == 0 or prec_chol.rows != prec_chol.cols:
        raise DimensionMismatchError.error(
            "non-empty square matrix (rows == cols)",
            "shape ("
            + String(prec_chol.rows)
            + ", "
            + String(prec_chol.cols)
            + ")",
            "_compute_precisions_from_cholesky_tied",
        )
    var D = prec_chol.rows
    var prec = Matrix[dtype](D, D, 0)
    for r in range(D):
        for c in range(r, D):
            var s: Float64 = 0.0
            for m in range(c, D):
                s += Float64(prec_chol[r, m]) * Float64(prec_chol[c, m])
            var val = Scalar[dtype](s)
            prec[r, c] = val
            prec[c, r] = val
    return prec^


def _compute_precisions_from_cholesky_full[
    dtype: DType
](prec_chol: List[Matrix[dtype]]) raises -> List[Matrix[dtype]]:
    """Recover full precision matrices P_k P_k^T for each component."""
    var K = len(prec_chol)
    if K == 0:
        raise DimensionMismatchError.error(
            "non-empty prec_chol list",
            "len == 0",
            "_compute_precisions_from_cholesky_full",
        )
    var precs = List[Matrix[dtype]](capacity=K)
    for k in range(K):
        precs.append(_compute_precisions_from_cholesky_tied(prec_chol[k]))
    return precs^


def _compute_precisions_from_cholesky_diag[
    dtype: DType
](prec_chol: Matrix[dtype]) raises -> Matrix[dtype]:
    """Recover diagonal precision entries by squaring precision Cholesky entries.
    """
    if prec_chol.rows == 0 or prec_chol.cols == 0:
        raise DimensionMismatchError.error(
            "non-empty prec_chol matrix",
            "shape ("
            + String(prec_chol.rows)
            + ", "
            + String(prec_chol.cols)
            + ")",
            "_compute_precisions_from_cholesky_diag",
        )
    var K = prec_chol.rows
    var D = prec_chol.cols
    var prec = Matrix[dtype](K, D, 0)
    for k in range(K):
        for j in range(D):
            var p = Float64(prec_chol[k, j])
            prec[k, j] = Scalar[dtype](p * p)
    return prec^


def _compute_precisions_from_cholesky_spherical[
    dtype: DType
](prec_chol: List[Scalar[dtype]]) raises -> List[Scalar[dtype]]:
    """Recover spherical precision entries by squaring precision Cholesky entries.
    """
    var K = len(prec_chol)
    if K == 0:
        raise DimensionMismatchError.error(
            "non-empty prec_chol list",
            "len == 0",
            "_compute_precisions_from_cholesky_spherical",
        )
    var prec = List[Scalar[dtype]](capacity=K)
    for k in range(K):
        var p = Float64(prec_chol[k])
        prec.append(Scalar[dtype](p * p))
    return prec^


def _compute_log_det_cholesky_tied[
    dtype: DType
](prec_chol: Matrix[dtype]) raises -> Float64:
    """Compute log(det(P)) = 0.5 * log(det(Sigma^-1)) for a square factor P."""
    if prec_chol.rows == 0 or prec_chol.rows != prec_chol.cols:
        raise DimensionMismatchError.error(
            "non-empty square matrix (rows == cols)",
            "shape ("
            + String(prec_chol.rows)
            + ", "
            + String(prec_chol.cols)
            + ")",
            "_compute_log_det_cholesky_tied",
        )
    var D = prec_chol.rows
    var log_det: Float64 = 0.0
    for j in range(D):
        var d = Float64(prec_chol[j, j])
        _check_positive_finite(d, "prec_chol", "_compute_log_det_cholesky_tied")
        log_det += log(d)
    return log_det


def _compute_log_det_cholesky_full[
    dtype: DType
](prec_chol: List[Matrix[dtype]]) raises -> List[Float64]:
    """Compute per-component log(det(P_k)) for full precision Cholesky factors.
    """
    var K = len(prec_chol)
    if K == 0:
        raise DimensionMismatchError.error(
            "non-empty prec_chol list",
            "len == 0",
            "_compute_log_det_cholesky_full",
        )
    var log_dets = List[Float64](capacity=K)
    for k in range(K):
        log_dets.append(_compute_log_det_cholesky_tied(prec_chol[k]))
    return log_dets^


def _compute_log_det_cholesky_diag[
    dtype: DType
](prec_chol: Matrix[dtype]) raises -> List[Float64]:
    """Compute per-component log(det(P_k)) for diagonal precision Cholesky factors.
    """
    if prec_chol.rows == 0 or prec_chol.cols == 0:
        raise DimensionMismatchError.error(
            "non-empty prec_chol matrix",
            "shape ("
            + String(prec_chol.rows)
            + ", "
            + String(prec_chol.cols)
            + ")",
            "_compute_log_det_cholesky_diag",
        )
    var K = prec_chol.rows
    var D = prec_chol.cols
    var log_dets = List[Float64](capacity=K)
    for k in range(K):
        var s: Float64 = 0.0
        for j in range(D):
            var d = Float64(prec_chol[k, j])
            _check_positive_finite(
                d, "prec_chol", "_compute_log_det_cholesky_diag"
            )
            s += log(d)
        log_dets.append(s)
    return log_dets^


def _compute_log_det_cholesky_spherical[
    dtype: DType
](prec_chol: List[Scalar[dtype]], n_features: Int) raises -> List[Float64]:
    """Compute per-component log(det(P_k)) for spherical precision Cholesky factors.
    """
    var K = len(prec_chol)
    if K == 0 or n_features <= 0:
        raise DimensionMismatchError.error(
            "K > 0 and n_features > 0",
            "K=" + String(K) + ", n_features=" + String(n_features),
            "_compute_log_det_cholesky_spherical",
        )
    var log_dets = List[Float64](capacity=K)
    var d_f64 = Float64(n_features)
    for k in range(K):
        var d = Float64(prec_chol[k])
        _check_positive_finite(
            d, "prec_chol", "_compute_log_det_cholesky_spherical"
        )
        log_dets.append(d_f64 * log(d))
    return log_dets^


def _validate_log_prob_inputs[
    dtype: DType
](X: Matrix[dtype], means: Matrix[dtype], context: String) raises:
    """Validate sample and component mean matrix shapes for density evaluation.
    """
    if X.rows == 0 or X.cols == 0:
        raise DimensionMismatchError.error(
            "non-empty X",
            "shape (" + String(X.rows) + ", " + String(X.cols) + ")",
            context,
        )
    if means.rows == 0 or means.cols != X.cols:
        raise DimensionMismatchError.error(
            "means shape (K > 0, " + String(X.cols) + ")",
            "shape (" + String(means.rows) + ", " + String(means.cols) + ")",
            context,
        )


def _estimate_log_gaussian_prob_full[
    dtype: DType
](
    X: Matrix[dtype],
    means: Matrix[dtype],
    prec_chol: List[Matrix[dtype]],
) raises -> Matrix[dtype]:
    """Evaluate component log Gaussian densities for full covariance matrices.
    """
    _validate_log_prob_inputs(X, means, "_estimate_log_gaussian_prob_full")
    var N = X.rows
    var D = X.cols
    var K = means.rows
    if len(prec_chol) != K:
        raise DimensionMismatchError.error(
            "len(prec_chol) == " + String(K),
            "len(prec_chol) == " + String(len(prec_chol)),
            "_estimate_log_gaussian_prob_full",
        )
    for k in range(K):
        if prec_chol[k].rows != D or prec_chol[k].cols != D:
            raise DimensionMismatchError.error(
                "prec_chol[k] shape (" + String(D) + ", " + String(D) + ")",
                "shape ("
                + String(prec_chol[k].rows)
                + ", "
                + String(prec_chol[k].cols)
                + ")",
                "_estimate_log_gaussian_prob_full",
            )

    var log_dets = _compute_log_det_cholesky_full(prec_chol)
    var log_prob = Matrix[dtype](N, K, 0)
    var diff = List[Float64](capacity=D)
    for _ in range(D):
        diff.append(0.0)
    var d_ln2pi = Float64(D) * LN_2PI

    for k in range(K):
        var log_det_k = log_dets[k]
        for i in range(N):
            for r in range(D):
                diff[r] = Float64(X[i, r]) - Float64(means[k, r])
            var mahal_sq: Float64 = 0.0
            for c in range(D):
                var y_c: Float64 = 0.0
                for r in range(c + 1):
                    y_c += diff[r] * Float64(prec_chol[k][r, c])
                mahal_sq += y_c * y_c
            log_prob[i, k] = Scalar[dtype](
                -0.5 * (d_ln2pi + mahal_sq) + log_det_k
            )

    return log_prob^


def _estimate_log_gaussian_prob_tied[
    dtype: DType
](
    X: Matrix[dtype],
    means: Matrix[dtype],
    prec_chol: Matrix[dtype],
) raises -> Matrix[dtype]:
    """Evaluate component log Gaussian densities for a tied covariance matrix.
    """
    _validate_log_prob_inputs(X, means, "_estimate_log_gaussian_prob_tied")
    var N = X.rows
    var D = X.cols
    var K = means.rows
    if prec_chol.rows != D or prec_chol.cols != D:
        raise DimensionMismatchError.error(
            "prec_chol shape (" + String(D) + ", " + String(D) + ")",
            "shape ("
            + String(prec_chol.rows)
            + ", "
            + String(prec_chol.cols)
            + ")",
            "_estimate_log_gaussian_prob_tied",
        )

    var log_det = _compute_log_det_cholesky_tied(prec_chol)
    var log_prob = Matrix[dtype](N, K, 0)
    var diff = List[Float64](capacity=D)
    for _ in range(D):
        diff.append(0.0)
    var d_ln2pi = Float64(D) * LN_2PI

    for k in range(K):
        for i in range(N):
            for r in range(D):
                diff[r] = Float64(X[i, r]) - Float64(means[k, r])
            var mahal_sq: Float64 = 0.0
            for c in range(D):
                var y_c: Float64 = 0.0
                for r in range(c + 1):
                    y_c += diff[r] * Float64(prec_chol[r, c])
                mahal_sq += y_c * y_c
            log_prob[i, k] = Scalar[dtype](
                -0.5 * (d_ln2pi + mahal_sq) + log_det
            )

    return log_prob^


def _estimate_log_gaussian_prob_diag[
    dtype: DType
](
    X: Matrix[dtype],
    means: Matrix[dtype],
    prec_chol: Matrix[dtype],
) raises -> Matrix[dtype]:
    """Evaluate component log Gaussian densities for diagonal covariances."""
    _validate_log_prob_inputs(X, means, "_estimate_log_gaussian_prob_diag")
    var N = X.rows
    var D = X.cols
    var K = means.rows
    if prec_chol.rows != K or prec_chol.cols != D:
        raise DimensionMismatchError.error(
            "prec_chol shape (" + String(K) + ", " + String(D) + ")",
            "shape ("
            + String(prec_chol.rows)
            + ", "
            + String(prec_chol.cols)
            + ")",
            "_estimate_log_gaussian_prob_diag",
        )

    var log_dets = _compute_log_det_cholesky_diag(prec_chol)
    var log_prob = Matrix[dtype](N, K, 0)
    var d_ln2pi = Float64(D) * LN_2PI

    for k in range(K):
        var log_det_k = log_dets[k]
        for i in range(N):
            var mahal_sq: Float64 = 0.0
            for j in range(D):
                var d = (Float64(X[i, j]) - Float64(means[k, j])) * Float64(
                    prec_chol[k, j]
                )
                mahal_sq += d * d
            log_prob[i, k] = Scalar[dtype](
                -0.5 * (d_ln2pi + mahal_sq) + log_det_k
            )

    return log_prob^


def _estimate_log_gaussian_prob_spherical[
    dtype: DType
](
    X: Matrix[dtype],
    means: Matrix[dtype],
    prec_chol: List[Scalar[dtype]],
) raises -> Matrix[dtype]:
    """Evaluate component log Gaussian densities for spherical covariances."""
    _validate_log_prob_inputs(X, means, "_estimate_log_gaussian_prob_spherical")
    var N = X.rows
    var D = X.cols
    var K = means.rows
    if len(prec_chol) != K:
        raise DimensionMismatchError.error(
            "len(prec_chol) == " + String(K),
            "len(prec_chol) == " + String(len(prec_chol)),
            "_estimate_log_gaussian_prob_spherical",
        )

    var log_dets = _compute_log_det_cholesky_spherical(prec_chol, D)
    var log_prob = Matrix[dtype](N, K, 0)
    var d_ln2pi = Float64(D) * LN_2PI

    for k in range(K):
        var p_k = Float64(prec_chol[k])
        var p_k2 = p_k * p_k
        var log_det_k = log_dets[k]
        for i in range(N):
            var sq_dist: Float64 = 0.0
            for j in range(D):
                var d = Float64(X[i, j]) - Float64(means[k, j])
                sq_dist += d * d
            var mahal_sq = sq_dist * p_k2
            log_prob[i, k] = Scalar[dtype](
                -0.5 * (d_ln2pi + mahal_sq) + log_det_k
            )

    return log_prob^


def _estimate_log_prob_resp[
    dtype: DType
](log_prob: Matrix[dtype], weights: List[Scalar[dtype]]) raises -> EStepResult[
    dtype
]:
    """Compute marginal log-probabilities, log-responsibilities, and mean bound.
    """
    if log_prob.rows == 0 or log_prob.cols == 0:
        raise DimensionMismatchError.error(
            "non-empty log_prob",
            "shape ("
            + String(log_prob.rows)
            + ", "
            + String(log_prob.cols)
            + ")",
            "_estimate_log_prob_resp",
        )
    var N = log_prob.rows
    var K = log_prob.cols
    if len(weights) != K:
        raise DimensionMismatchError.error(
            "len(weights) == " + String(K),
            "len(weights) == " + String(len(weights)),
            "_estimate_log_prob_resp",
        )

    var log_weights = List[Float64](capacity=K)
    var has_positive_weight = False
    for k in range(K):
        var w = Float64(weights[k])
        if isnan(w) or isinf(w) or w < 0.0:
            raise InvalidParameterError.error(
                "weights",
                "Mixture weights must be finite and non-negative, got "
                + String(w),
            )
        if w > 1e-300:
            has_positive_weight = True
            log_weights.append(log(w))
        else:
            log_weights.append(-1e30)
    if not has_positive_weight:
        raise InvalidParameterError.error(
            "weights", "At least one mixture weight must be positive"
        )

    var log_prob_norm = List[Scalar[dtype]](capacity=N)
    var log_resp = Matrix[dtype](N, K, 0)
    var weighted = List[Float64](capacity=K)
    for _ in range(K):
        weighted.append(0.0)

    var total_log_prob: Float64 = 0.0
    for i in range(N):
        var max_val: Float64 = 0.0
        for k in range(K):
            var lp = Float64(log_prob[i, k])
            if isnan(lp) or isinf(lp):
                raise InvalidParameterError.error(
                    "log_prob",
                    "Log-probability matrix contains NaN or infinite values",
                )
            var wlp = lp + log_weights[k]
            weighted[k] = wlp
            if k == 0 or wlp > max_val:
                max_val = wlp

        var sum_exp: Float64 = 0.0
        for k in range(K):
            sum_exp += exp(weighted[k] - max_val)
        var lse = max_val + log(sum_exp)

        log_prob_norm.append(Scalar[dtype](lse))
        total_log_prob += lse
        for k in range(K):
            log_resp[i, k] = Scalar[dtype](weighted[k] - lse)

    var mean_lp = total_log_prob / Float64(N)
    return EStepResult[dtype](log_prob_norm^, log_resp^, mean_lp)


def _prng_uniform_open01(mut rng: PRNG) -> Float64:
    """Draw a pseudo-random uniform Float64 strictly inside (0.0, 1.0)."""
    return (Float64(rng.next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0)


def _prng_standard_normal_pair(mut rng: PRNG) -> Tuple[Float64, Float64]:
    """Generate two independent standard normal N(0, 1) variates via Box-Muller.
    """
    var u1 = _prng_uniform_open01(rng)
    var u2 = _prng_uniform_open01(rng)
    var radius = sqrt(-2.0 * log(u1))
    var angle = TWO_PI * u2
    return (radius * cos(angle), radius * sin(angle))


def _sample_multivariate_normal[
    dtype: DType
](
    mut rng: PRNG,
    mean: List[Scalar[dtype]],
    cov_chol_lower: Matrix[dtype],
) raises -> List[Scalar[dtype]]:
    """Sample a vector from N(mean, L L^T) given lower Cholesky factor L."""
    var D = len(mean)
    if D == 0 or cov_chol_lower.rows != D or cov_chol_lower.cols != D:
        raise DimensionMismatchError.error(
            "cov_chol_lower shape (" + String(D) + ", " + String(D) + ")",
            "shape ("
            + String(cov_chol_lower.rows)
            + ", "
            + String(cov_chol_lower.cols)
            + ")",
            "_sample_multivariate_normal",
        )

    var z = List[Float64](capacity=D)
    var i = 0
    while i < D:
        var pair = _prng_standard_normal_pair(rng)
        z.append(pair[0])
        if i + 1 < D:
            z.append(pair[1])
        i += 2

    var out_vec = List[Scalar[dtype]](capacity=D)
    for r in range(D):
        var acc = Float64(mean[r])
        for c in range(r + 1):
            acc += Float64(cov_chol_lower[r, c]) * z[c]
        out_vec.append(Scalar[dtype](acc))

    return out_vec^


def _sample_categorical_counts(
    mut rng: PRNG, weights: List[Float64], n_samples: Int
) raises -> List[Int]:
    """Sample multinomial component counts for n_samples independent draws."""
    var K = len(weights)
    if K == 0:
        raise DimensionMismatchError.error(
            "non-empty weights", "len == 0", "_sample_categorical_counts"
        )
    if n_samples < 0:
        raise InvalidParameterError.error(
            "n_samples",
            "n_samples must be non-negative, got " + String(n_samples),
        )

    var total: Float64 = 0.0
    for k in range(K):
        var w = weights[k]
        if isnan(w) or isinf(w) or w < 0.0:
            raise InvalidParameterError.error(
                "weights", "Weights must be finite and non-negative"
            )
        total += w
    if total <= 0.0:
        raise InvalidParameterError.error(
            "weights", "Sum of weights must be strictly positive"
        )

    var cdf = List[Float64](capacity=K)
    var cum: Float64 = 0.0
    for k in range(K):
        cum += weights[k] / total
        cdf.append(cum)
    cdf[K - 1] = 1.0

    var counts = List[Int](capacity=K)
    for _ in range(K):
        counts.append(0)

    for _ in range(n_samples):
        var u = _prng_uniform_open01(rng)
        var chosen = K - 1
        for k in range(K):
            if u <= cdf[k]:
                chosen = k
                break
        counts[chosen] += 1

    return counts^


def _n_parameters(
    n_components: Int, n_features: Int, covariance_type: String
) raises -> Int:
    """Return total number of free parameters for AIC/BIC model selection."""
    if n_components < 1:
        raise InvalidParameterError.error(
            "n_components",
            "n_components must be >= 1, got " + String(n_components),
        )
    if n_features < 1:
        raise InvalidParameterError.error(
            "n_features",
            "n_features must be >= 1, got " + String(n_features),
        )

    var cov_params: Int
    if covariance_type == "full":
        cov_params = n_components * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        cov_params = n_components
    else:
        raise InvalidParameterError.error(
            "covariance_type",
            "covariance_type must be 'full', 'tied', 'diag', or 'spherical',"
            " got '"
            + covariance_type
            + "'",
        )

    var mean_params = n_features * n_components
    var weight_params = n_components - 1
    return cov_params + mean_params + weight_params
