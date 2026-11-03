from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..base.estimator import Regressor
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)
from ._kernels import (
    compute_gamma,
    compute_gram_matrix,
    compute_kernel_matrix,
)
from ._smo import _kernel_svr_solver, SMOResult


struct SVR[compute_dtype: DType = DType.float64](
    Copyable, Movable, Regressor, Serializable
):
    """Epsilon-Support Vector Regression using non-linear kernels.

    Fits an epsilon-insensitive regression model via dual coordinate descent
    across kernel mappings (RBF, Polynomial, Sigmoid, and Linear):

    $$
    \\min_{\\beta} \\frac{1}{2} \\beta^T Q \\beta - y^T \\beta + \\epsilon \\sum_{i=1}^n |\\beta_i|
    \\quad \\text{s.t.} \\quad -C \\le \\beta_i \\le C
    $$

    Parameters:
        compute_dtype: Floating point precision for kernel operations. Default DType.float64.

    Args:
        C: Regularization parameter. Must be strictly positive. Default 1.0.
        epsilon: Epsilon-tube width within which no penalty is associated in the
            training loss function. Must be non-negative. Default 0.1.
        kernel: Kernel routine ('rbf', 'linear', 'poly', 'sigmoid'). Default 'rbf'.
        degree: Degree of the polynomial kernel function. Default 3.
        gamma: Kernel coefficient mode ('scale' or 'auto'). Default 'scale'.
        gamma_val: Explicit numeric gamma. Overrides gamma mode when > 0. Default -1.0.
        coef0: Independent term in poly and sigmoid kernels. Default 0.0.
        tol: Tolerance for stopping criterion. Default 1e-3.
        fit_intercept: Whether to estimate an intercept term. Default True.
        max_iter: Hard limit on solver iterations. Default 10000.

    Attributes:
        support_vectors_: Support vectors of shape (n_support_vectors, n_features).
        support_indices_: Indices of support vectors in training set.
        dual_coef_: Dual coefficients for support vectors of length n_support_vectors.
        intercept_: Bias term in decision function.
        n_features_in_: Number of features seen during fit.
        n_iter_: Number of iterations run by the solver.
        gamma_: Resolved kernel scale coefficient.
        is_fitted: Whether the estimator has been fitted.

    Examples:
        ```mojo
        from strata.svm import SVR
        from strata.core import Matrix

        var reg = SVR[DType.float64](C=1.0, epsilon=0.1, kernel="rbf")
        reg.fit(X_train, y_train)
        var preds = reg.predict(X_test)
        ```
    """

    var C: Float64
    var epsilon: Float64
    var kernel: String
    var degree: Int
    var gamma: String
    var gamma_val: Float64
    var coef0: Float64
    var tol: Float64
    var fit_intercept: Bool
    var max_iter: Int

    var support_vectors_: Matrix[Self.compute_dtype]
    var support_indices_: List[Int]
    var dual_coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]
    var n_features_in_: Int
    var n_iter_: Int
    var gamma_: Scalar[Self.compute_dtype]
    var is_fitted: Bool

    def __init__(
        out self,
        C: Float64 = 1.0,
        epsilon: Float64 = 0.1,
        kernel: String = "rbf",
        degree: Int = 3,
        gamma: String = "scale",
        gamma_val: Float64 = -1.0,
        coef0: Float64 = 0.0,
        tol: Float64 = 1e-3,
        fit_intercept: Bool = True,
        max_iter: Int = 10000,
    ):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.gamma_val = gamma_val
        self.coef0 = coef0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

        self.support_vectors_ = Matrix[Self.compute_dtype](0, 0, 0.0)
        self.support_indices_ = List[Int]()
        self.dual_coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = Scalar[Self.compute_dtype](0.0)
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.gamma_ = Scalar[Self.compute_dtype](0.0)
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        self.C = copy.C
        self.epsilon = copy.epsilon
        self.kernel = copy.kernel
        self.degree = copy.degree
        self.gamma = copy.gamma
        self.gamma_val = copy.gamma_val
        self.coef0 = copy.coef0
        self.tol = copy.tol
        self.fit_intercept = copy.fit_intercept
        self.max_iter = copy.max_iter

        self.support_vectors_ = copy.support_vectors_.copy()
        self.support_indices_ = copy.support_indices_.copy()
        self.dual_coef_ = copy.dual_coef_.copy()
        self.intercept_ = copy.intercept_
        self.n_features_in_ = copy.n_features_in_
        self.n_iter_ = copy.n_iter_
        self.gamma_ = copy.gamma_
        self.is_fitted = copy.is_fitted

    def __init__(out self, *, deinit move: Self):
        self.C = move.C
        self.epsilon = move.epsilon
        self.kernel = move.kernel
        self.degree = move.degree
        self.gamma = move.gamma
        self.gamma_val = move.gamma_val
        self.coef0 = move.coef0
        self.tol = move.tol
        self.fit_intercept = move.fit_intercept
        self.max_iter = move.max_iter

        self.support_vectors_ = move.support_vectors_^
        self.support_indices_ = move.support_indices_^
        self.dual_coef_ = move.dual_coef_^
        self.intercept_ = move.intercept_
        self.n_features_in_ = move.n_features_in_
        self.n_iter_ = move.n_iter_
        self.gamma_ = move.gamma_
        self.is_fitted = move.is_fitted

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fits the kernel Support Vector Regressor on features X and targets y.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target values of length n_samples.

        Raises:
            InvalidParameterError: If hyperparameters are invalid.
            DimensionMismatchError: If X and y sample counts differ.
        """
        check_X_y[feat_dtype, target_dtype](X, y)
        check_floating_dtype[Self.compute_dtype]()

        if self.C <= 0:
            raise InvalidParameterError.error(
                "C", "C must be strictly positive"
            )
        if self.epsilon < 0:
            raise InvalidParameterError.error(
                "epsilon", "epsilon must be non-negative"
            )
        if self.tol <= 0:
            raise InvalidParameterError.error(
                "tol", "tol must be strictly positive"
            )
        if self.degree < 1:
            raise InvalidParameterError.error("degree", "degree must be >= 1")
        if self.max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter", "max_iter must be strictly positive"
            )
        if (
            self.kernel != "rbf"
            and self.kernel != "linear"
            and self.kernel != "poly"
            and self.kernel != "sigmoid"
        ):
            raise InvalidParameterError.error(
                "kernel",
                "Unsupported kernel '"
                + self.kernel
                + "'. Expected 'rbf', 'linear', 'poly', or 'sigmoid'.",
            )

        var n = X.rows
        var d = X.cols
        self.n_features_in_ = d

        var X_comp = X.cast[Self.compute_dtype]()
        var y_comp = List[Scalar[Self.compute_dtype]](capacity=n)
        for i in range(n):
            y_comp.append(Scalar[Self.compute_dtype](y[i]))

        var gamma_val_scalar = Scalar[Self.compute_dtype](self.gamma_val)
        self.gamma_ = compute_gamma(X_comp, self.gamma, gamma_val_scalar)

        var coef0_scalar = Scalar[Self.compute_dtype](self.coef0)
        var C_scalar = Scalar[Self.compute_dtype](self.C)
        var eps_scalar = Scalar[Self.compute_dtype](self.epsilon)
        var tol_scalar = Scalar[Self.compute_dtype](self.tol)

        var Gram = compute_gram_matrix(
            X_comp, self.kernel, self.gamma_, self.degree, coef0_scalar
        )

        var res = _kernel_svr_solver(
            Gram,
            y_comp,
            C_scalar,
            eps_scalar,
            self.fit_intercept,
            tol_scalar,
            self.max_iter,
        )

        var n_sv = len(res.support_indices)
        self.support_indices_ = res.support_indices.copy()
        self.support_vectors_ = Matrix[Self.compute_dtype](n_sv, d, 0.0)
        for m in range(n_sv):
            var sv_idx = res.support_indices[m]
            for j in range(d):
                self.support_vectors_[m, j] = X_comp[sv_idx, j]

        self.dual_coef_ = res.dual_coef.copy()
        self.intercept_ = res.intercept
        self.n_iter_ = res.n_iter
        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fits the regressor using a Dataset container."""
        self.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predict continuous target values for samples in X.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            List[Scalar[feat_dtype]]: Predicted target values of length n_samples.

        Raises:
            NotFittedError: If model is not fitted.
            DimensionMismatchError: If feature count does not match n_features_in_.
        """
        check_is_fitted("SVR", self.is_fitted)
        check_array[feat_dtype](X)

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "SVR.predict",
            )

        var n = X.rows
        var s = self.support_vectors_.rows
        var preds = List[Scalar[feat_dtype]](capacity=n)

        if s == 0:
            var b_out = Scalar[feat_dtype](self.intercept_)
            for _ in range(n):
                preds.append(b_out)
            return preds^

        var X_comp = X.cast[Self.compute_dtype]()
        var coef0_scalar = Scalar[Self.compute_dtype](self.coef0)

        # Cache (n x s) kernel evaluations against support vectors
        var K_eval = compute_kernel_matrix(
            X_comp,
            self.support_vectors_,
            self.kernel,
            self.gamma_,
            self.degree,
            coef0_scalar,
        )

        for i in range(n):
            var dot: Scalar[Self.compute_dtype] = 0.0
            for m in range(s):
                dot += K_eval[i, m] * self.dual_coef_[m]
            preds.append(Scalar[feat_dtype](dot + self.intercept_))

        return preds^

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[
        Scalar[feat_dtype]
    ]:
        """Predicts target values for a Dataset container."""
        return self.predict[feat_dtype](dataset.records)

    def serialize(self, mut writer: BufferWriter):
        """Serializes SVR parameters and fitted state into BufferWriter."""
        write_header(writer, "SVR")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.C)
        writer.write_float64(self.epsilon)
        writer.write_string(self.kernel)
        writer.write_int(self.degree)
        writer.write_string(self.gamma)
        writer.write_float64(self.gamma_val)
        writer.write_float64(self.coef0)
        writer.write_float64(self.tol)
        writer.write_bool(self.fit_intercept)
        writer.write_int(self.max_iter)
        writer.write_int(self.n_features_in_)
        writer.write_int(self.n_iter_)
        writer.write_float64(Float64(self.gamma_))
        writer.write_int_list(self.support_indices_)
        writer.write_matrix[Self.compute_dtype](self.support_vectors_)
        writer.write_float_list[Self.compute_dtype](self.dual_coef_)
        writer.write_float64(Float64(self.intercept_))

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes SVR from BufferReader."""
        check_header(reader, "SVR")
        var is_fitted = reader.read_bool()
        var C = reader.read_float64()
        var epsilon = reader.read_float64()
        var kernel = reader.read_string()
        var degree = reader.read_int()
        var gamma = reader.read_string()
        var gamma_val = reader.read_float64()
        var coef0 = reader.read_float64()
        var tol = reader.read_float64()
        var fit_intercept = reader.read_bool()
        var max_iter = reader.read_int()
        var n_features_in_ = reader.read_int()
        var n_iter_ = reader.read_int()
        var gamma_ = Scalar[Self.compute_dtype](reader.read_float64())
        var support_indices_ = reader.read_int_list()
        var support_vectors_ = reader.read_matrix[Self.compute_dtype]()
        var dual_coef_ = reader.read_float_list[Self.compute_dtype]()
        var intercept_ = Scalar[Self.compute_dtype](reader.read_float64())

        var model = Self(
            C=C,
            epsilon=epsilon,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            gamma_val=gamma_val,
            coef0=coef0,
            tol=tol,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.n_iter_ = n_iter_
        model.gamma_ = gamma_
        model.support_indices_ = support_indices_^
        model.support_vectors_ = support_vectors_^
        model.dual_coef_ = dual_coef_^
        model.intercept_ = intercept_
        return model^
