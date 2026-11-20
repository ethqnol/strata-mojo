from std.math import abs, exp, log, sqrt, isnan, isinf
from ..base.estimator import Clusterer
from ..core.matrix import Matrix
from ..cluster.kmeans import KMeans
from ..cluster._common import (
    _init_centroids_kmeans_plus_plus,
    _init_centroids_random,
    _assign_nearest_and_inertia,
)
from ..utils.random import PRNG
from ..utils.validation import (
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..exceptions.errors import (
    InvalidParameterError,
    DimensionMismatchError,
    DataConversionError,
)
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)
from ._gmm_math import (
    EStepResult,
    _estimate_gaussian_nk_and_means,
    _estimate_covariances_full,
    _estimate_covariances_tied,
    _estimate_covariances_diag,
    _estimate_covariances_spherical,
    _cholesky_lower,
    _precision_cholesky_from_lower,
    _compute_precision_cholesky_full,
    _compute_precision_cholesky_tied,
    _compute_precision_cholesky_diag,
    _compute_precision_cholesky_spherical,
    _compute_precisions_from_cholesky_full,
    _compute_precisions_from_cholesky_tied,
    _compute_precisions_from_cholesky_diag,
    _compute_precisions_from_cholesky_spherical,
    _estimate_log_gaussian_prob_full,
    _estimate_log_gaussian_prob_tied,
    _estimate_log_gaussian_prob_diag,
    _estimate_log_gaussian_prob_spherical,
    _estimate_log_prob_resp,
    _prng_uniform_open01,
    _prng_standard_normal_pair,
    _sample_multivariate_normal,
    _sample_categorical_counts,
    _n_parameters,
)


struct GaussianSample[dtype: DType = DType.float64](Copyable, Movable):
    """Generated synthetic samples and component labels from GaussianMixture.sample.

    Parameters:
        dtype: Scalar floating-point precision type. Default DType.float64.

    Attributes:
        X: Sampled observations matrix of shape $(N, D)$.
        y: Originating mixture component index for each sample of length $N$.
    """

    var X: Matrix[Self.dtype]
    var y: List[Int]

    def __init__(out self, var X: Matrix[Self.dtype], var y: List[Int]):
        """Initialize a GaussianSample container."""
        self.X = X^
        self.y = y^

    def __init__(out self, *, copy: Self):
        """Copy an existing GaussianSample container."""
        self.X = copy.X.copy()
        self.y = copy.y.copy()

    def __init__(out self, *, deinit move: Self):
        """Move an existing GaussianSample container."""
        self.X = move.X^
        self.y = move.y^


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


def _copy_matrix_list[
    dtype: DType
](src: List[Matrix[dtype]]) -> List[Matrix[dtype]]:
    """Deep-copy a list of dense Matrix containers."""
    var out = List[Matrix[dtype]](capacity=len(src))
    for i in range(len(src)):
        out.append(src[i].copy())
    return out^


def _list_to_col_matrix[
    dtype: DType
](vals: List[Scalar[dtype]]) -> Matrix[dtype]:
    """Pack a 1D scalar list into a (len, 1) Matrix."""
    var n = len(vals)
    var mat = Matrix[dtype](n, 1, 0)
    for i in range(n):
        mat[i, 0] = vals[i]
    return mat^


def _col_matrix_to_list[
    dtype: DType
](mat: Matrix[dtype]) -> List[Scalar[dtype]]:
    """Extract column 0 from a (K, 1) Matrix into a 1D scalar list."""
    var n = mat.rows
    var vals = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        vals.append(mat[i, 0])
    return vals^


def _validate_gmm_hyperparams(
    n_components: Int,
    covariance_type: String,
    tol: Float64,
    reg_covar: Float64,
    max_iter: Int,
    n_init: Int,
    init_params: String,
) raises:
    """Validate GaussianMixture hyperparameters."""
    if n_components < 1:
        raise InvalidParameterError.error(
            "n_components",
            "n_components must be >= 1, got " + String(n_components),
        )
    if (
        covariance_type != "full"
        and covariance_type != "tied"
        and covariance_type != "diag"
        and covariance_type != "spherical"
    ):
        raise InvalidParameterError.error(
            "covariance_type",
            "covariance_type must be 'full', 'tied', 'diag', or 'spherical',"
            " got '"
            + covariance_type
            + "'",
        )
    if isnan(tol) or isinf(tol) or tol < 0.0:
        raise InvalidParameterError.error(
            "tol", "tol must be finite and >= 0.0, got " + String(tol)
        )
    if isnan(reg_covar) or isinf(reg_covar) or reg_covar < 0.0:
        raise InvalidParameterError.error(
            "reg_covar",
            "reg_covar must be finite and >= 0.0, got " + String(reg_covar),
        )
    if max_iter < 1:
        raise InvalidParameterError.error(
            "max_iter", "max_iter must be >= 1, got " + String(max_iter)
        )
    if n_init < 1:
        raise InvalidParameterError.error(
            "n_init", "n_init must be >= 1, got " + String(n_init)
        )
    if (
        init_params != "kmeans"
        and init_params != "k-means++"
        and init_params != "random"
        and init_params != "random_from_data"
    ):
        raise InvalidParameterError.error(
            "init_params",
            "init_params must be 'kmeans', 'k-means++', 'random', or"
            " 'random_from_data', got '"
            + init_params
            + "'",
        )


def _validate_weights_init[
    dtype: DType
](weights_init: List[Scalar[dtype]], K: Int) raises:
    """Validate user-provided initial mixture weights."""
    if len(weights_init) != K:
        raise InvalidParameterError.error(
            "weights_init",
            "weights_init length ("
            + String(len(weights_init))
            + ") must match n_components ("
            + String(K)
            + ")",
        )
    var w_sum: Float64 = 0.0
    var has_pos = False
    for k in range(K):
        var w = Float64(weights_init[k])
        if isnan(w) or isinf(w) or w < 0.0:
            raise InvalidParameterError.error(
                "weights_init",
                "weights_init entries must be finite and >= 0.0",
            )
        if w > 0.0:
            has_pos = True
        w_sum += w
    if not has_pos:
        raise InvalidParameterError.error(
            "weights_init", "weights_init must contain a positive weight"
        )
    var diff_from_one = abs(w_sum - 1.0)
    if diff_from_one > 1e-3:
        raise InvalidParameterError.error(
            "weights_init",
            "Sum of weights_init must equal 1.0, got " + String(w_sum),
        )


def _validate_means_init[
    dtype: DType
](means_init: Matrix[dtype], K: Int, D: Int) raises:
    """Validate user-provided initial component means."""
    if means_init.rows != K or means_init.cols != D:
        raise DimensionMismatchError.error(
            "means_init shape (" + String(K) + ", " + String(D) + ")",
            "shape ("
            + String(means_init.rows)
            + ", "
            + String(means_init.cols)
            + ")",
            "GaussianMixture.fit",
        )
    for k in range(K):
        for j in range(D):
            var v = Float64(means_init[k, j])
            if isnan(v) or isinf(v):
                raise InvalidParameterError.error(
                    "means_init",
                    "means_init contains NaN or infinite values",
                )


def _invert_spd_precision[
    dtype: DType
](P_init: Matrix[dtype]) raises -> Matrix[dtype]:
    """Validate an SPD precision matrix Lambda and return Sigma = Lambda^-1."""
    var D = P_init.rows
    for r in range(D):
        for c in range(r, D):
            var a = Float64(P_init[r, c])
            var b = Float64(P_init[c, r])
            if isnan(a) or isinf(a) or isnan(b) or isinf(b):
                raise InvalidParameterError.error(
                    "precisions_init",
                    "precisions_init contains NaN or infinite values",
                )
            if abs(a - b) > 1e-6 * (1.0 + abs(a)):
                raise InvalidParameterError.error(
                    "precisions_init",
                    "precisions_init matrices must be symmetric",
                )
    var L_lam = _cholesky_lower(P_init)
    var Q = _precision_cholesky_from_lower(L_lam)
    return _compute_precisions_from_cholesky_tied(Q)


def _compute_log_gaussian_prob[
    dtype: DType
](
    X: Matrix[dtype],
    means: Matrix[dtype],
    precisions_cholesky: List[Matrix[dtype]],
    covariance_type: String,
) raises -> Matrix[dtype]:
    """Dispatch component log-Gaussian density evaluation by covariance type."""
    if covariance_type == "full":
        return _estimate_log_gaussian_prob_full(X, means, precisions_cholesky)
    elif covariance_type == "tied":
        return _estimate_log_gaussian_prob_tied(
            X, means, precisions_cholesky[0]
        )
    elif covariance_type == "diag":
        return _estimate_log_gaussian_prob_diag(
            X, means, precisions_cholesky[0]
        )
    else:
        return _estimate_log_gaussian_prob_spherical(
            X, means, _col_matrix_to_list(precisions_cholesky[0])
        )


def _compute_precisions[
    dtype: DType
](
    precisions_cholesky: List[Matrix[dtype]], covariance_type: String
) raises -> List[Matrix[dtype]]:
    """Recover precision matrices from stored Cholesky factors."""
    var precs = List[Matrix[dtype]]()
    if covariance_type == "full":
        return _compute_precisions_from_cholesky_full(precisions_cholesky)
    elif covariance_type == "tied":
        precs.append(
            _compute_precisions_from_cholesky_tied(precisions_cholesky[0])
        )
        return precs^
    elif covariance_type == "diag":
        precs.append(
            _compute_precisions_from_cholesky_diag(precisions_cholesky[0])
        )
        return precs^
    else:
        var sph_prec = _compute_precisions_from_cholesky_spherical(
            _col_matrix_to_list(precisions_cholesky[0])
        )
        precs.append(_list_to_col_matrix(sph_prec))
        return precs^


def _m_step_from_resp[
    dtype: DType
](
    X_comp: Matrix[dtype],
    resp: Matrix[dtype],
    covariance_type: String,
    reg_covar: Float64,
) raises -> _GMMState[dtype]:
    """Execute one M-step from responsibility matrix resp."""
    var stats = _estimate_gaussian_nk_and_means(X_comp, resp)
    var K = len(stats.nk)
    var total_nk: Float64 = 0.0
    for k in range(K):
        total_nk += Float64(stats.nk[k])
    var inv_total = 1.0 / total_nk
    var weights = List[Scalar[dtype]](capacity=K)
    for k in range(K):
        weights.append(Scalar[dtype](Float64(stats.nk[k]) * inv_total))

    var covariances = List[Matrix[dtype]]()
    var precisions_cholesky = List[Matrix[dtype]]()

    if covariance_type == "full":
        covariances = _estimate_covariances_full(
            X_comp, resp, stats.nk, stats.means, reg_covar
        )
        precisions_cholesky = _compute_precision_cholesky_full(covariances)
    elif covariance_type == "tied":
        var cov_t = _estimate_covariances_tied(
            X_comp, resp, stats.nk, stats.means, reg_covar
        )
        var pchol_t = _compute_precision_cholesky_tied(cov_t)
        covariances.append(cov_t^)
        precisions_cholesky.append(pchol_t^)
    elif covariance_type == "diag":
        var cov_d = _estimate_covariances_diag(
            X_comp, resp, stats.nk, stats.means, reg_covar
        )
        var pchol_d = _compute_precision_cholesky_diag(cov_d)
        covariances.append(cov_d^)
        precisions_cholesky.append(pchol_d^)
    else:
        var cov_s_list = _estimate_covariances_spherical(
            X_comp, resp, stats.nk, stats.means, reg_covar
        )
        var pchol_s_list = _compute_precision_cholesky_spherical(cov_s_list)
        covariances.append(_list_to_col_matrix(cov_s_list))
        precisions_cholesky.append(_list_to_col_matrix(pchol_s_list))

    return _GMMState[dtype](
        weights^, stats.means.copy(), covariances^, precisions_cholesky^
    )


def _validate_fitted_matrix_shapes[
    dtype: DType
](
    mats: List[Matrix[dtype]],
    K: Int,
    D: Int,
    covariance_type: String,
    field_name: String,
) raises:
    """Validate serialized covariance/precision matrix list shapes."""
    if covariance_type == "full":
        if len(mats) != K:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: "
                + field_name
                + " length mismatch"
            )
        for k in range(K):
            if mats[k].rows != D or mats[k].cols != D:
                raise DataConversionError.error(
                    "Invalid serialized GaussianMixture: "
                    + field_name
                    + " matrix shape mismatch"
                )
    elif covariance_type == "tied":
        if len(mats) != 1 or mats[0].rows != D or mats[0].cols != D:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: "
                + field_name
                + " tied shape mismatch"
            )
    elif covariance_type == "diag":
        if len(mats) != 1 or mats[0].rows != K or mats[0].cols != D:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: "
                + field_name
                + " diag shape mismatch"
            )
    else:
        if len(mats) != 1 or mats[0].rows != K or mats[0].cols != 1:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: "
                + field_name
                + " spherical shape mismatch"
            )


struct GaussianMixture[compute_dtype: DType = DType.float64](
    Clusterer, Copyable, Movable, Serializable
):
    """Gaussian Mixture Model trained with Expectation-Maximization (EM).

    Models a distribution as a weighted mixture of $K$ multivariate Gaussian
    components:

    $$
    p(x) = \\sum_{k=1}^{K} \\pi_k \\, \\mathcal{N}(x \\mid \\mu_k, \\Sigma_k)
    $$

    supporting `'full'`, `'tied'`, `'diag'`, and `'spherical'` covariance
    structures with multi-start initialization, custom parameter priors,
    density scoring, AIC/BIC model selection, and generative sampling.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        n_components: Number of mixture components ($K \\ge 1$). Default 1.
        covariance_type: Covariance parameterization ('full', 'tied', 'diag',
            'spherical'). Default 'full'.
        tol: Convergence threshold on EM log-likelihood lower bound change.
            Default 1e-3.
        reg_covar: Non-negative regularization added to covariance diagonals.
            Default 1e-6.
        max_iter: Maximum number of EM iterations per initialization. Default 100.
        n_init: Number of independent initializations to evaluate. Default 1.
        init_params: Initialization strategy ('kmeans', 'k-means++', 'random',
            'random_from_data'). Default 'kmeans'.
        weights_init: Optional user-provided initial weights of length $K$.
            Default empty.
        means_init: Optional user-provided initial means of shape $(K, D)$.
            Default empty.
        precisions_init: Optional user-provided initial precisions matching
            `covariance_type` shape conventions. Default empty.
        random_state: PRNG seed for deterministic initialization and sampling.
            Default 42.
        warm_start: If True, reuse previous fit parameters as initialization.
            Default False.

    Attributes:
        weights_: Fitted mixture weights of length $K$.
        means_: Fitted component centers of shape $(K, D)$.
        covariances_: Fitted covariances stored per `covariance_type`.
        precisions_: Fitted precision matrices ($\\Sigma_k^{-1}$).
        precisions_cholesky_: Upper-triangular (or reciprocal std) Cholesky factors
            of the precision matrices.
        converged_: True if EM reached tolerance within `max_iter`.
        n_iter_: Number of EM iterations executed in the winning restart.
        lower_bound_: Final mean log-likelihood lower bound achieved on training data.
        n_features_in_: Number of features observed during fit.
        is_fitted: Boolean flag indicating whether the model has been fitted.

    Examples:
        ```mojo
        from strata.mixture import GaussianMixture
        from strata.core import Matrix

        var gmm = GaussianMixture[DType.float64](
            n_components=3, covariance_type="full", random_state=42
        )
        gmm.fit(X_train)
        var labels = gmm.predict(X_test)
        var probs = gmm.predict_proba(X_test)
        ```
    """

    var n_components: Int
    var covariance_type: String
    var tol: Float64
    var reg_covar: Float64
    var max_iter: Int
    var n_init: Int
    var init_params: String
    var weights_init: List[Scalar[Self.compute_dtype]]
    var means_init: Matrix[Self.compute_dtype]
    var precisions_init: List[Matrix[Self.compute_dtype]]
    var random_state: Int
    var warm_start: Bool

    var weights_: List[Scalar[Self.compute_dtype]]
    var means_: Matrix[Self.compute_dtype]
    var covariances_: List[Matrix[Self.compute_dtype]]
    var precisions_: List[Matrix[Self.compute_dtype]]
    var precisions_cholesky_: List[Matrix[Self.compute_dtype]]
    var converged_: Bool
    var n_iter_: Int
    var lower_bound_: Float64
    var n_features_in_: Int
    var is_fitted: Bool

    def __init__(
        out self,
        n_components: Int = 1,
        covariance_type: String = "full",
        tol: Float64 = 1e-3,
        reg_covar: Float64 = 1e-6,
        max_iter: Int = 100,
        n_init: Int = 1,
        init_params: String = "kmeans",
        weights_init: List[Scalar[Self.compute_dtype]] = List[
            Scalar[Self.compute_dtype]
        ](),
        means_init: Matrix[Self.compute_dtype] = Matrix[Self.compute_dtype](
            0, 0, 0
        ),
        precisions_init: List[Matrix[Self.compute_dtype]] = List[
            Matrix[Self.compute_dtype]
        ](),
        random_state: Int = 42,
        warm_start: Bool = False,
    ) raises:
        """Initialize the GaussianMixture model.

        Args:
            n_components: Number of Gaussian components (>= 1). Default 1.
            covariance_type: Covariance structure ('full', 'tied', 'diag', 'spherical').
                Default 'full'.
            tol: EM convergence tolerance (>= 0.0). Default 1e-3.
            reg_covar: Diagonal covariance regularization (>= 0.0). Default 1e-6.
            max_iter: Maximum EM iterations per restart (>= 1). Default 100.
            n_init: Number of initialization restarts (>= 1). Default 1.
            init_params: Initialization method ('kmeans', 'k-means++', 'random',
                'random_from_data'). Default 'kmeans'.
            weights_init: Optional initial mixture weights. Default empty.
            means_init: Optional initial component means. Default empty.
            precisions_init: Optional initial precision matrices. Default empty.
            random_state: PRNG seed. Default 42.
            warm_start: Reuse fitted state as initialization. Default False.

        Raises:
            InvalidParameterError: If any hyperparameter is out of domain.
        """
        check_floating_dtype[Self.compute_dtype, "GaussianMixture"]()
        _validate_gmm_hyperparams(
            n_components,
            covariance_type,
            tol,
            reg_covar,
            max_iter,
            n_init,
            init_params,
        )

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init.copy()
        self.means_init = means_init.copy()
        self.precisions_init = _copy_matrix_list(precisions_init)
        self.random_state = random_state
        self.warm_start = warm_start

        self.weights_ = List[Scalar[Self.compute_dtype]]()
        self.means_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.covariances_ = List[Matrix[Self.compute_dtype]]()
        self.precisions_ = List[Matrix[Self.compute_dtype]]()
        self.precisions_cholesky_ = List[Matrix[Self.compute_dtype]]()
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = 0.0
        self.n_features_in_ = 0
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        """Copy an existing GaussianMixture instance."""
        self.n_components = copy.n_components
        self.covariance_type = copy.covariance_type
        self.tol = copy.tol
        self.reg_covar = copy.reg_covar
        self.max_iter = copy.max_iter
        self.n_init = copy.n_init
        self.init_params = copy.init_params
        self.weights_init = copy.weights_init.copy()
        self.means_init = copy.means_init.copy()
        self.precisions_init = _copy_matrix_list(copy.precisions_init)
        self.random_state = copy.random_state
        self.warm_start = copy.warm_start

        self.weights_ = copy.weights_.copy()
        self.means_ = copy.means_.copy()
        self.covariances_ = _copy_matrix_list(copy.covariances_)
        self.precisions_ = _copy_matrix_list(copy.precisions_)
        self.precisions_cholesky_ = _copy_matrix_list(copy.precisions_cholesky_)
        self.converged_ = copy.converged_
        self.n_iter_ = copy.n_iter_
        self.lower_bound_ = copy.lower_bound_
        self.n_features_in_ = copy.n_features_in_
        self.is_fitted = copy.is_fitted

    def __init__(out self, *, deinit move: Self):
        """Move an existing GaussianMixture instance."""
        self.n_components = move.n_components
        self.covariance_type = move.covariance_type^
        self.tol = move.tol
        self.reg_covar = move.reg_covar
        self.max_iter = move.max_iter
        self.n_init = move.n_init
        self.init_params = move.init_params^
        self.weights_init = move.weights_init^
        self.means_init = move.means_init^
        self.precisions_init = move.precisions_init^
        self.random_state = move.random_state
        self.warm_start = move.warm_start

        self.weights_ = move.weights_^
        self.means_ = move.means_^
        self.covariances_ = move.covariances_^
        self.precisions_ = move.precisions_^
        self.precisions_cholesky_ = move.precisions_cholesky_^
        self.converged_ = move.converged_
        self.n_iter_ = move.n_iter_
        self.lower_bound_ = move.lower_bound_
        self.n_features_in_ = move.n_features_in_
        self.is_fitted = move.is_fitted

    def copy(self) -> Self:
        """Return a deep copy of the GaussianMixture instance."""
        return Self(copy=self)

    def _init_responsibilities(
        self, X_comp: Matrix[Self.compute_dtype], run_seed: Int
    ) raises -> Matrix[Self.compute_dtype]:
        """Generate initial responsibility matrix of shape (N, K)."""
        var N = X_comp.rows
        var K = self.n_components
        var resp = Matrix[Self.compute_dtype](N, K, 0)

        if K == 1:
            for i in range(N):
                resp[i, 0] = 1
            return resp^

        if self.init_params == "kmeans":
            var km = KMeans[Self.compute_dtype](
                n_clusters=K,
                init="k-means++",
                n_init=1,
                max_iter=300,
                tol=1e-4,
                random_state=run_seed,
            )
            km.fit(X_comp)
            for i in range(N):
                resp[i, km.labels_[i]] = 1
        elif (
            self.init_params == "k-means++"
            or self.init_params == "random_from_data"
        ):
            var centers = _init_centroids_kmeans_plus_plus(
                X_comp, K, run_seed
            ) if self.init_params == "k-means++" else _init_centroids_random(
                X_comp, K, run_seed
            )
            var labels = List[Int](capacity=N)
            var min_sq = List[Scalar[Self.compute_dtype]](capacity=N)
            for _ in range(N):
                labels.append(0)
                min_sq.append(0)
            _ = _assign_nearest_and_inertia(X_comp, centers, labels, min_sq)
            for i in range(N):
                resp[i, labels[i]] = 1
        else:
            var rng = PRNG(run_seed)
            for i in range(N):
                var row_sum: Float64 = 0.0
                for k in range(K):
                    var u = _prng_uniform_open01(rng)
                    resp[i, k] = Scalar[Self.compute_dtype](u)
                    row_sum += u
                var inv_sum = 1.0 / row_sum
                for k in range(K):
                    resp[i, k] = Scalar[Self.compute_dtype](
                        Float64(resp[i, k]) * inv_sum
                    )

        return resp^

    def _initialize_params(
        self, X_comp: Matrix[Self.compute_dtype], run_seed: Int
    ) raises -> _GMMState[Self.compute_dtype]:
        """Initialize mixture parameters with optional user overrides."""
        var K = self.n_components
        var D = X_comp.cols

        var resp = self._init_responsibilities(X_comp, run_seed)
        var state = _m_step_from_resp(
            X_comp, resp, self.covariance_type, self.reg_covar
        )

        if len(self.weights_init) > 0:
            _validate_weights_init(self.weights_init, K)
            state.weights = self.weights_init.copy()

        if self.means_init.rows > 0 or self.means_init.cols > 0:
            _validate_means_init(self.means_init, K, D)
            state.means = self.means_init.copy()

        if len(self.precisions_init) > 0:
            if self.covariance_type == "full":
                if len(self.precisions_init) != K:
                    raise InvalidParameterError.error(
                        "precisions_init",
                        "precisions_init length ("
                        + String(len(self.precisions_init))
                        + ") must match n_components ("
                        + String(K)
                        + ") for covariance_type='full'",
                    )
                var covs = List[Matrix[Self.compute_dtype]](capacity=K)
                var pchols = List[Matrix[Self.compute_dtype]](capacity=K)
                for k in range(K):
                    if (
                        self.precisions_init[k].rows != D
                        or self.precisions_init[k].cols != D
                    ):
                        raise DimensionMismatchError.error(
                            "precisions_init["
                            + String(k)
                            + "] shape ("
                            + String(D)
                            + ", "
                            + String(D)
                            + ")",
                            "shape ("
                            + String(self.precisions_init[k].rows)
                            + ", "
                            + String(self.precisions_init[k].cols)
                            + ")",
                            "GaussianMixture.fit",
                        )
                    var cov_k = _invert_spd_precision(self.precisions_init[k])
                    var pchol_k = _compute_precision_cholesky_tied(cov_k)
                    covs.append(cov_k^)
                    pchols.append(pchol_k^)
                state.covariances = covs^
                state.precisions_cholesky = pchols^
            elif self.covariance_type == "tied":
                if (
                    len(self.precisions_init) != 1
                    or self.precisions_init[0].rows != D
                    or self.precisions_init[0].cols != D
                ):
                    raise DimensionMismatchError.error(
                        "precisions_init of 1 matrix of shape ("
                        + String(D)
                        + ", "
                        + String(D)
                        + ")",
                        "len=" + String(len(self.precisions_init)),
                        "GaussianMixture.fit",
                    )
                var cov_t = _invert_spd_precision(self.precisions_init[0])
                var pchol_t = _compute_precision_cholesky_tied(cov_t)
                var covs = List[Matrix[Self.compute_dtype]](capacity=1)
                var pchols = List[Matrix[Self.compute_dtype]](capacity=1)
                covs.append(cov_t^)
                pchols.append(pchol_t^)
                state.covariances = covs^
                state.precisions_cholesky = pchols^
            elif self.covariance_type == "diag":
                if (
                    len(self.precisions_init) != 1
                    or self.precisions_init[0].rows != K
                    or self.precisions_init[0].cols != D
                ):
                    raise DimensionMismatchError.error(
                        "precisions_init of 1 matrix of shape ("
                        + String(K)
                        + ", "
                        + String(D)
                        + ")",
                        "len=" + String(len(self.precisions_init)),
                        "GaussianMixture.fit",
                    )
                var cov_d = Matrix[Self.compute_dtype](K, D, 0)
                var pchol_d = Matrix[Self.compute_dtype](K, D, 0)
                for k in range(K):
                    for j in range(D):
                        var v = Float64(self.precisions_init[0][k, j])
                        if isnan(v) or isinf(v) or v <= 0.0:
                            raise InvalidParameterError.error(
                                "precisions_init",
                                (
                                    "Diagonal precisions_init entries must be"
                                    " finite and > 0.0"
                                ),
                            )
                        cov_d[k, j] = Scalar[Self.compute_dtype](1.0 / v)
                        pchol_d[k, j] = Scalar[Self.compute_dtype](sqrt(v))
                var covs = List[Matrix[Self.compute_dtype]](capacity=1)
                var pchols = List[Matrix[Self.compute_dtype]](capacity=1)
                covs.append(cov_d^)
                pchols.append(pchol_d^)
                state.covariances = covs^
                state.precisions_cholesky = pchols^
            else:
                if (
                    len(self.precisions_init) != 1
                    or self.precisions_init[0].rows != K
                    or self.precisions_init[0].cols != 1
                ):
                    raise DimensionMismatchError.error(
                        "precisions_init of 1 matrix of shape ("
                        + String(K)
                        + ", 1)",
                        "len=" + String(len(self.precisions_init)),
                        "GaussianMixture.fit",
                    )
                var cov_s = Matrix[Self.compute_dtype](K, 1, 0)
                var pchol_s = Matrix[Self.compute_dtype](K, 1, 0)
                for k in range(K):
                    var v = Float64(self.precisions_init[0][k, 0])
                    if isnan(v) or isinf(v) or v <= 0.0:
                        raise InvalidParameterError.error(
                            "precisions_init",
                            (
                                "Spherical precisions_init entries must be"
                                " finite and > 0.0"
                            ),
                        )
                    cov_s[k, 0] = Scalar[Self.compute_dtype](1.0 / v)
                    pchol_s[k, 0] = Scalar[Self.compute_dtype](sqrt(v))
                var covs = List[Matrix[Self.compute_dtype]](capacity=1)
                var pchols = List[Matrix[Self.compute_dtype]](capacity=1)
                covs.append(cov_s^)
                pchols.append(pchol_s^)
                state.covariances = covs^
                state.precisions_cholesky = pchols^

        return state^

    def _e_step_raw[
        in_dtype: DType
    ](
        self,
        X: Matrix[in_dtype],
        weights: List[Scalar[Self.compute_dtype]],
        means: Matrix[Self.compute_dtype],
        precisions_cholesky: List[Matrix[Self.compute_dtype]],
    ) raises -> EStepResult[Self.compute_dtype]:
        """Evaluate E-step on Matrix X using explicit mixture parameters."""
        var X_comp = X.cast[Self.compute_dtype]()
        var log_prob = _compute_log_gaussian_prob(
            X_comp, means, precisions_cholesky, self.covariance_type
        )
        return _estimate_log_prob_resp(log_prob, weights)

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fit Gaussian Mixture Model parameters using Expectation-Maximization.

        Args:
            X: Training feature matrix of shape $(N, D)$ with $N \\ge K$.

        Raises:
            InvalidParameterError: If hyperparameters or X entries are invalid.
            DimensionMismatchError: If X is empty, $N < K$, or feature count
                mismatches on warm_start.
        """
        check_array[in_dtype](X)
        _validate_gmm_hyperparams(
            self.n_components,
            self.covariance_type,
            self.tol,
            self.reg_covar,
            self.max_iter,
            self.n_init,
            self.init_params,
        )

        var N = X.rows
        var D = X.cols
        var K = self.n_components
        if N < K:
            raise DimensionMismatchError.error(
                "n_samples >= n_components (" + String(K) + ")",
                "n_samples == " + String(N),
                "GaussianMixture.fit",
            )

        var do_init = not (self.warm_start and self.is_fitted)
        if not do_init:
            if D != self.n_features_in_:
                raise DimensionMismatchError.error(
                    "X.cols == " + String(self.n_features_in_),
                    "X.cols == " + String(D),
                    "GaussianMixture.fit (warm_start)",
                )
            if len(self.weights_) != K or self.means_.rows != K:
                raise InvalidParameterError.error(
                    "n_components",
                    (
                        "Cannot change n_components when warm_start=True after"
                        " fitting"
                    ),
                )
            var cov_ok: Bool
            if self.covariance_type == "full":
                cov_ok = (
                    len(self.covariances_) == K
                    and len(self.precisions_cholesky_) == K
                    and self.covariances_[0].rows == D
                    and self.covariances_[0].cols == D
                )
            elif self.covariance_type == "tied":
                cov_ok = (
                    len(self.covariances_) == 1
                    and len(self.precisions_cholesky_) == 1
                    and self.covariances_[0].rows == D
                    and self.covariances_[0].cols == D
                )
            elif self.covariance_type == "diag":
                cov_ok = (
                    len(self.covariances_) == 1
                    and len(self.precisions_cholesky_) == 1
                    and self.covariances_[0].rows == K
                    and self.covariances_[0].cols == D
                )
            else:
                cov_ok = (
                    len(self.covariances_) == 1
                    and len(self.precisions_cholesky_) == 1
                    and self.covariances_[0].rows == K
                    and self.covariances_[0].cols == 1
                )
            if not cov_ok:
                raise InvalidParameterError.error(
                    "covariance_type",
                    (
                        "Cannot change covariance_type when warm_start=True"
                        " after fitting"
                    ),
                )

        var X_comp = X.cast[Self.compute_dtype]()
        var n_init_runs = self.n_init if do_init else 1

        var best_lower_bound: Float64 = -1e308
        var best_n_iter = 0
        var best_converged = False
        var best_state = _GMMState[Self.compute_dtype](
            List[Scalar[Self.compute_dtype]](),
            Matrix[Self.compute_dtype](0, 0, 0),
            List[Matrix[Self.compute_dtype]](),
            List[Matrix[Self.compute_dtype]](),
        )
        var has_best = False

        var resp = Matrix[Self.compute_dtype](N, K, 0)

        for init_idx in range(n_init_runs):
            var state: _GMMState[Self.compute_dtype]
            if do_init:
                var run_seed = self.random_state + init_idx * 97
                state = self._initialize_params(X_comp, run_seed)
            else:
                state = _GMMState[Self.compute_dtype](
                    self.weights_.copy(),
                    self.means_.copy(),
                    _copy_matrix_list(self.covariances_),
                    _copy_matrix_list(self.precisions_cholesky_),
                )

            var prev_lower_bound: Float64 = 0.0
            var converged = False
            var n_iter = 0

            for it in range(1, self.max_iter + 1):
                var log_prob = _compute_log_gaussian_prob(
                    X_comp,
                    state.means,
                    state.precisions_cholesky,
                    self.covariance_type,
                )
                var estep = _estimate_log_prob_resp(log_prob, state.weights)
                var lower_bound = estep.mean_log_prob

                for i in range(N):
                    for k in range(K):
                        resp[i, k] = Scalar[Self.compute_dtype](
                            exp(Float64(estep.log_resp[i, k]))
                        )

                state = _m_step_from_resp(
                    X_comp, resp, self.covariance_type, self.reg_covar
                )
                n_iter = it

                if it > 1 and abs(lower_bound - prev_lower_bound) < self.tol:
                    converged = True
                    break

                prev_lower_bound = lower_bound

            var final_lp = _compute_log_gaussian_prob(
                X_comp,
                state.means,
                state.precisions_cholesky,
                self.covariance_type,
            )
            var final_estep = _estimate_log_prob_resp(final_lp, state.weights)
            var final_lb = final_estep.mean_log_prob

            if not has_best or final_lb > best_lower_bound:
                has_best = True
                best_lower_bound = final_lb
                best_n_iter = n_iter
                best_converged = converged
                best_state = state^

        var best_precisions = _compute_precisions(
            best_state.precisions_cholesky, self.covariance_type
        )

        # Commit validated fitted state atomically
        self.weights_ = best_state.weights.copy()
        self.means_ = best_state.means.copy()
        self.covariances_ = _copy_matrix_list(best_state.covariances)
        self.precisions_cholesky_ = _copy_matrix_list(
            best_state.precisions_cholesky
        )
        self.precisions_ = best_precisions^
        self.converged_ = best_converged
        self.n_iter_ = best_n_iter
        self.lower_bound_ = best_lower_bound
        self.n_features_in_ = D
        self.is_fitted = True

    def _validate_predict_input[
        in_dtype: DType
    ](self, X: Matrix[in_dtype], context: String) raises:
        """Validate fitted status and input matrix shape for inference."""
        check_is_fitted("GaussianMixture", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                context,
            )

    def predict[in_dtype: DType](self, X: Matrix[in_dtype]) raises -> List[Int]:
        """Predict component labels for observations in X.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            List[Int]: Most probable component index per sample of length $N$.

        Raises:
            NotFittedError: If model is not fitted.
            DimensionMismatchError: If X is empty or feature count mismatches.
        """
        self._validate_predict_input(X, "GaussianMixture.predict")
        var estep = self._e_step_raw(
            X, self.weights_, self.means_, self.precisions_cholesky_
        )
        var N = estep.log_resp.rows
        var K = estep.log_resp.cols
        var preds = List[Int](capacity=N)

        for i in range(N):
            var best_k = 0
            var best_val = estep.log_resp[i, 0]
            for k in range(1, K):
                if estep.log_resp[i, k] > best_val:
                    best_val = estep.log_resp[i, k]
                    best_k = k
            preds.append(best_k)

        return preds^

    def fit_predict[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> List[Int]:
        """Fit the GaussianMixture on X and predict component assignments.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            List[Int]: Predicted component indices of length $N$.
        """
        self.fit[in_dtype](X)
        return self.predict[in_dtype](X)

    def predict_proba[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Predict posterior responsibility probabilities for each component.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Matrix[Self.compute_dtype]: Posterior probability matrix of shape
                $(N, K)$ with each row summing to 1.0.

        Raises:
            NotFittedError: If model is not fitted.
            DimensionMismatchError: If X is empty or feature count mismatches.
        """
        self._validate_predict_input(X, "GaussianMixture.predict_proba")
        var estep = self._e_step_raw(
            X, self.weights_, self.means_, self.precisions_cholesky_
        )
        var N = estep.log_resp.rows
        var K = estep.log_resp.cols
        var probs = Matrix[Self.compute_dtype](N, K, 0)

        for i in range(N):
            var row_sum: Float64 = 0.0
            for k in range(K):
                var p = exp(Float64(estep.log_resp[i, k]))
                row_sum += p
                probs[i, k] = Scalar[Self.compute_dtype](p)
            if row_sum > 0.0:
                var inv_sum = 1.0 / row_sum
                for k in range(K):
                    probs[i, k] = Scalar[Self.compute_dtype](
                        Float64(probs[i, k]) * inv_sum
                    )

        return probs^

    def score_samples[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> List[Scalar[Self.compute_dtype]]:
        """Compute per-sample marginal log-likelihood $\\ln p(x_i)$.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            List[Scalar[Self.compute_dtype]]: Marginal log-densities of length $N$.

        Raises:
            NotFittedError: If model is not fitted.
            DimensionMismatchError: If X is empty or feature count mismatches.
        """
        self._validate_predict_input(X, "GaussianMixture.score_samples")
        var estep = self._e_step_raw(
            X, self.weights_, self.means_, self.precisions_cholesky_
        )
        return estep.log_prob_norm.copy()

    def score[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Scalar[Self.compute_dtype]:
        """Compute the average per-sample log-likelihood over X.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Scalar[Self.compute_dtype]: Mean marginal log-likelihood across samples.

        Raises:
            NotFittedError: If model is not fitted.
            DimensionMismatchError: If X is empty or feature count mismatches.
        """
        self._validate_predict_input(X, "GaussianMixture.score")
        var estep = self._e_step_raw(
            X, self.weights_, self.means_, self.precisions_cholesky_
        )
        return Scalar[Self.compute_dtype](estep.mean_log_prob)

    def aic[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Scalar[Self.compute_dtype]:
        """Compute the Akaike Information Criterion (AIC) on observations X.

        $$
        \\text{AIC} = -2 \\ln L + 2 p
        $$

        where $p$ is the number of free parameters. Lower values indicate better
        penalized model fit.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Scalar[Self.compute_dtype]: Evaluated AIC value.
        """
        var mean_ll = Float64(self.score(X))
        var n_params = _n_parameters(
            self.n_components, self.n_features_in_, self.covariance_type
        )
        return Scalar[Self.compute_dtype](
            -2.0 * mean_ll * Float64(X.rows) + 2.0 * Float64(n_params)
        )

    def bic[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Scalar[Self.compute_dtype]:
        """Compute the Bayesian Information Criterion (BIC) on observations X.

        $$
        \\text{BIC} = -2 \\ln L + p \\ln N
        $$

        where $p$ is the number of free parameters. Lower values indicate better
        penalized model fit.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Scalar[Self.compute_dtype]: Evaluated BIC value.
        """
        var mean_ll = Float64(self.score(X))
        var n_params = _n_parameters(
            self.n_components, self.n_features_in_, self.covariance_type
        )
        return Scalar[Self.compute_dtype](
            -2.0 * mean_ll * Float64(X.rows)
            + Float64(n_params) * log(Float64(X.rows))
        )

    def sample(
        self, n_samples: Int = 1, random_state: Int = -1
    ) raises -> GaussianSample[Self.compute_dtype]:
        """Generate random samples from the fitted Gaussian mixture distribution.

        Args:
            n_samples: Number of samples to draw ($\\ge 1$). Default 1.
            random_state: Optional PRNG seed override. When < 0, uses model's
                `random_state`. Default -1.

        Returns:
            GaussianSample[Self.compute_dtype]: Container holding sampled matrix
                `X` of shape $(n\\_samples, D)$ and component indices `y`.

        Raises:
            NotFittedError: If model is not fitted.
            InvalidParameterError: If `n_samples < 1`.
        """
        check_is_fitted("GaussianMixture", self.is_fitted)
        if n_samples < 1:
            raise InvalidParameterError.error(
                "n_samples",
                "n_samples must be >= 1, got " + String(n_samples),
            )

        var seed = random_state if random_state >= 0 else self.random_state
        var rng = PRNG(seed)
        var K = self.n_components
        var D = self.n_features_in_

        var weights_f64 = List[Float64](capacity=K)
        for k in range(K):
            weights_f64.append(Float64(self.weights_[k]))
        var counts = _sample_categorical_counts(rng, weights_f64, n_samples)

        var X_out = Matrix[Self.compute_dtype](n_samples, D, 0)
        var y_out = List[Int](capacity=n_samples)
        var row_idx = 0

        if self.covariance_type == "full" or self.covariance_type == "tied":
            var chols = List[Matrix[Self.compute_dtype]]()
            if self.covariance_type == "full":
                for k in range(K):
                    chols.append(_cholesky_lower(self.covariances_[k]))
            else:
                chols.append(_cholesky_lower(self.covariances_[0]))

            for k in range(K):
                var chol_k = (
                    chols[k].copy() if self.covariance_type
                    == "full" else chols[0].copy()
                )
                var mean_k = List[Scalar[Self.compute_dtype]](capacity=D)
                for j in range(D):
                    mean_k.append(self.means_[k, j])

                for _ in range(counts[k]):
                    var drawn = _sample_multivariate_normal(rng, mean_k, chol_k)
                    for j in range(D):
                        X_out[row_idx, j] = drawn[j]
                    y_out.append(k)
                    row_idx += 1
        else:
            var stds = Matrix[DType.float64](K, D, 0.0)
            for k in range(K):
                if self.covariance_type == "diag":
                    for j in range(D):
                        stds[k, j] = sqrt(Float64(self.covariances_[0][k, j]))
                else:
                    var s = sqrt(Float64(self.covariances_[0][k, 0]))
                    for j in range(D):
                        stds[k, j] = s

            for k in range(K):
                for _ in range(counts[k]):
                    var j = 0
                    while j < D:
                        var pair = _prng_standard_normal_pair(rng)
                        X_out[row_idx, j] = Scalar[Self.compute_dtype](
                            Float64(self.means_[k, j]) + stds[k, j] * pair[0]
                        )
                        if j + 1 < D:
                            X_out[row_idx, j + 1] = Scalar[Self.compute_dtype](
                                Float64(self.means_[k, j + 1])
                                + stds[k, j + 1] * pair[1]
                            )
                        j += 2
                    y_out.append(k)
                    row_idx += 1

        return GaussianSample[Self.compute_dtype](X_out^, y_out^)

    def serialize(self, mut writer: BufferWriter):
        """Serialize GaussianMixture parameters and fitted state into BufferWriter.
        """
        write_header(writer, "GaussianMixture")
        writer.write_bool(self.is_fitted)
        writer.write_int(self.n_components)
        writer.write_string(self.covariance_type)
        writer.write_float64(self.tol)
        writer.write_float64(self.reg_covar)
        writer.write_int(self.max_iter)
        writer.write_int(self.n_init)
        writer.write_string(self.init_params)
        writer.write_float_list[Self.compute_dtype](self.weights_init)
        writer.write_matrix[Self.compute_dtype](self.means_init)

        writer.write_int(len(self.precisions_init))
        for i in range(len(self.precisions_init)):
            writer.write_matrix[Self.compute_dtype](self.precisions_init[i])

        writer.write_int(self.random_state)
        writer.write_bool(self.warm_start)

        writer.write_int(self.n_features_in_)
        writer.write_bool(self.converged_)
        writer.write_int(self.n_iter_)
        writer.write_float64(self.lower_bound_)
        writer.write_float_list[Self.compute_dtype](self.weights_)
        writer.write_matrix[Self.compute_dtype](self.means_)

        writer.write_int(len(self.covariances_))
        for i in range(len(self.covariances_)):
            writer.write_matrix[Self.compute_dtype](self.covariances_[i])

        writer.write_int(len(self.precisions_))
        for i in range(len(self.precisions_)):
            writer.write_matrix[Self.compute_dtype](self.precisions_[i])

        writer.write_int(len(self.precisions_cholesky_))
        for i in range(len(self.precisions_cholesky_)):
            writer.write_matrix[Self.compute_dtype](
                self.precisions_cholesky_[i]
            )

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserialize GaussianMixture from BufferReader."""
        check_header(reader, "GaussianMixture")
        var is_fitted = reader.read_bool()
        var n_components = reader.read_int()
        var covariance_type = reader.read_string()
        var tol = reader.read_float64()
        var reg_covar = reader.read_float64()
        var max_iter = reader.read_int()
        var n_init = reader.read_int()
        var init_params = reader.read_string()
        var weights_init = reader.read_float_list[Self.compute_dtype]()
        var means_init = reader.read_matrix[Self.compute_dtype]()

        var num_p_init = reader.read_int()
        if num_p_init < 0:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: negative precisions_init"
                " length"
            )
        var precisions_init = List[Matrix[Self.compute_dtype]](
            capacity=num_p_init
        )
        for _ in range(num_p_init):
            precisions_init.append(reader.read_matrix[Self.compute_dtype]())

        var random_state = reader.read_int()
        var warm_start = reader.read_bool()

        var n_features_in_ = reader.read_int()
        if n_features_in_ < 0:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: negative n_features_in_"
            )
        var converged_ = reader.read_bool()
        var n_iter_ = reader.read_int()
        if n_iter_ < 0:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: negative n_iter_"
            )
        var lower_bound_ = reader.read_float64()
        var weights_ = reader.read_float_list[Self.compute_dtype]()
        var means_ = reader.read_matrix[Self.compute_dtype]()

        var num_covs = reader.read_int()
        if num_covs < 0:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: negative covariances_"
                " length"
            )
        var covariances_ = List[Matrix[Self.compute_dtype]](capacity=num_covs)
        for _ in range(num_covs):
            covariances_.append(reader.read_matrix[Self.compute_dtype]())

        var num_precs = reader.read_int()
        if num_precs < 0:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: negative precisions_"
                " length"
            )
        var precisions_ = List[Matrix[Self.compute_dtype]](capacity=num_precs)
        for _ in range(num_precs):
            precisions_.append(reader.read_matrix[Self.compute_dtype]())

        var num_pchols = reader.read_int()
        if num_pchols < 0:
            raise DataConversionError.error(
                "Invalid serialized GaussianMixture: negative"
                " precisions_cholesky_ length"
            )
        var precisions_cholesky_ = List[Matrix[Self.compute_dtype]](
            capacity=num_pchols
        )
        for _ in range(num_pchols):
            precisions_cholesky_.append(
                reader.read_matrix[Self.compute_dtype]()
            )

        var model = Self(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
        )

        if is_fitted:
            if (
                n_features_in_ < 1
                or len(weights_) != n_components
                or means_.rows != n_components
                or means_.cols != n_features_in_
            ):
                raise DataConversionError.error(
                    "Invalid serialized GaussianMixture: inconsistent fitted"
                    " dimensions"
                )
            _validate_fitted_matrix_shapes(
                covariances_,
                n_components,
                n_features_in_,
                covariance_type,
                "covariances_",
            )
            _validate_fitted_matrix_shapes(
                precisions_,
                n_components,
                n_features_in_,
                covariance_type,
                "precisions_",
            )
            _validate_fitted_matrix_shapes(
                precisions_cholesky_,
                n_components,
                n_features_in_,
                covariance_type,
                "precisions_cholesky_",
            )
        else:
            if (
                n_features_in_ != 0
                or len(weights_) != 0
                or means_.rows != 0
                or means_.cols != 0
                or num_covs != 0
                or num_precs != 0
                or num_pchols != 0
            ):
                raise DataConversionError.error(
                    "Invalid serialized GaussianMixture: unfitted model has"
                    " non-empty state"
                )

        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.converged_ = converged_
        model.n_iter_ = n_iter_
        model.lower_bound_ = lower_bound_
        model.weights_ = weights_^
        model.means_ = means_^
        model.covariances_ = covariances_^
        model.precisions_ = precisions_^
        model.precisions_cholesky_ = precisions_cholesky_^
        return model^
