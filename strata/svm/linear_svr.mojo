from ..core.matrix import Matrix
from ..base.estimator import Regressor
from ..core.dataset import Dataset
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
from ._linear_svm_fast import _dual_coordinate_descent_svr


struct LinearSVR[compute_dtype: DType = DType.float64](
    Copyable, Movable, Regressor, Serializable
):
    """Linear Support Vector Regression.

    Fits an epsilon-insensitive linear model using L2 regularization and
    dual coordinate descent (LIBLINEAR algorithm, Ho & Lin 2012). Supports both
    standard epsilon-insensitive loss (L1-SVR) and squared epsilon-insensitive
    loss (L2-SVR).

    Parameters:
        compute_dtype: Precision for floating-point calculations. Default DType.float64.

    Args:
        epsilon: Epsilon parameter in the epsilon-insensitive loss function.
            The value of this parameter defines the margin of tolerance where
            no penalty is given to errors. Must be non-negative. Default 0.0.
        tol: Tolerance for stopping criterion. Default 1e-4.
        C: Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive. Default 1.0.
        loss: Specifies the loss function. 'epsilon_insensitive' corresponds to
            L1 loss; 'squared_epsilon_insensitive' corresponds to L2 loss.
            Default 'epsilon_insensitive'.
        fit_intercept: Whether to calculate the intercept for this model. Default True.
        intercept_scaling: When fit_intercept is True, instance vector x becomes
            [x, intercept_scaling]. Default 1.0.
        dual: Select the algorithm to solve the dual or primal optimization problem.
            Only True is currently supported. Default True.
        max_iter: The maximum number of coordinate descent iterations to run. Default 1000.
        random_state: Seed for pseudo-random number generator permutation. Default 42.

    Attributes:
        coef_: Weights assigned to the features of length n_features.
        intercept_: Constants in decision function.
        n_features_in_: Number of features seen during fit.
        n_iter_: Number of iterations run until convergence.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.svm import LinearSVR
        from strata.core import Matrix

        var X = Matrix[DType.float64](4, 2)
        X[0, 0] = 1.0; X[0, 1] = 2.0
        X[1, 0] = 2.0; X[1, 1] = 3.0
        X[2, 0] = 3.0; X[2, 1] = 4.0
        X[3, 0] = 4.0; X[3, 1] = 5.0

        var y: List[Scalar[DType.float64]] = [3.0, 5.0, 7.0, 9.0]
        var reg = LinearSVR(epsilon=0.0, C=1.0)
        reg.fit(X, y)
        var preds = reg.predict(X)
        ```
    """

    var epsilon: Float64
    var tol: Float64
    var C: Float64
    var loss: String
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var dual: Bool
    var max_iter: Int
    var random_state: Int

    var coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]
    var n_features_in_: Int
    var n_iter_: Int
    var is_fitted: Bool

    def __init__(
        out self,
        epsilon: Float64 = 0.0,
        tol: Float64 = 1e-4,
        C: Float64 = 1.0,
        loss: String = "epsilon_insensitive",
        fit_intercept: Bool = True,
        intercept_scaling: Float64 = 1.0,
        dual: Bool = True,
        max_iter: Int = 1000,
        random_state: Int = 42,
    ) raises:
        """Initialize the LinearSVR estimator.

        Args:
            epsilon: Epsilon tube width (must be non-negative). Default 0.0.
            tol: Convergence tolerance (must be strictly positive). Default 1e-4.
            C: Regularization strength (must be strictly positive). Default 1.0.
            loss: Loss function ('epsilon_insensitive' or 'squared_epsilon_insensitive'). Default 'epsilon_insensitive'.
            fit_intercept: Whether to compute bias offset. Default True.
            intercept_scaling: Synthetic feature coordinate multiplier. Default 1.0.
            dual: Whether to optimize dual formulation. Default True.
            max_iter: Maximum epochs. Default 1000.
            random_state: Random shuffle seed. Default 42.

        Raises:
            InvalidParameterError: If any parameter is invalid.
        """
        check_floating_dtype[Self.compute_dtype, "LinearSVR"]()

        if C <= 0.0:
            raise InvalidParameterError.error(
                "C",
                "C must be strictly positive, got " + String(C),
            )
        if epsilon < 0.0:
            raise InvalidParameterError.error(
                "epsilon",
                "epsilon must be non-negative, got " + String(epsilon),
            )
        if tol <= 0.0:
            raise InvalidParameterError.error(
                "tol",
                "tol must be strictly positive, got " + String(tol),
            )
        if max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter",
                "max_iter must be strictly positive, got " + String(max_iter),
            )
        if intercept_scaling <= 0.0:
            raise InvalidParameterError.error(
                "intercept_scaling",
                "intercept_scaling must be strictly positive, got "
                + String(intercept_scaling),
            )
        if (
            loss != "epsilon_insensitive"
            and loss != "squared_epsilon_insensitive"
            and loss != "l1"
            and loss != "l2"
        ):
            raise InvalidParameterError.error(
                "loss",
                "Unknown loss '"
                + loss
                + "'. Expected 'epsilon_insensitive' or"
                " 'squared_epsilon_insensitive'.",
            )
        if not dual:
            raise InvalidParameterError.error(
                "dual",
                "LinearSVR currently only supports dual=True.",
            )

        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.max_iter = max_iter
        self.random_state = random_state

        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = Scalar[Self.compute_dtype](0.0)
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        """Creates a deep copy of an existing LinearSVR instance."""
        self.epsilon = copy.epsilon
        self.tol = copy.tol
        self.C = copy.C
        self.loss = copy.loss
        self.fit_intercept = copy.fit_intercept
        self.intercept_scaling = copy.intercept_scaling
        self.dual = copy.dual
        self.max_iter = copy.max_iter
        self.random_state = copy.random_state

        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_
        self.n_features_in_ = copy.n_features_in_
        self.n_iter_ = copy.n_iter_
        self.is_fitted = copy.is_fitted

    def copy(self) -> Self:
        """Returns a deep copy of the LinearSVR instance."""
        return Self(copy=self)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit the Linear Support Vector Regressor on training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Continuous target values vector of length n_samples.

        Raises:
            DimensionMismatchError: If sample counts do not match or matrices are empty.
            InvalidParameterError: If data contains NaNs or Infs.
        """
        check_X_y(X, y)

        # Reset any previous fit state
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = Scalar[Self.compute_dtype](0.0)
        self.n_features_in_ = X.cols
        self.n_iter_ = 0
        self.is_fitted = False

        var n_samples = X.rows
        var n_features = X.cols

        # Convert X and y to compute_dtype
        var X_cast = Matrix[Self.compute_dtype](n_samples, n_features)
        for r in range(n_samples):
            for c in range(n_features):
                X_cast[r, c] = Scalar[Self.compute_dtype](X[r, c])

        var y_cast = List[Scalar[Self.compute_dtype]](capacity=n_samples)
        for i in range(n_samples):
            y_cast.append(Scalar[Self.compute_dtype](y[i]))

        var res = _dual_coordinate_descent_svr[Self.compute_dtype](
            X=X_cast,
            y=y_cast,
            C=Scalar[Self.compute_dtype](self.C),
            p=Scalar[Self.compute_dtype](self.epsilon),
            loss=self.loss,
            fit_intercept=self.fit_intercept,
            intercept_scaling=Scalar[Self.compute_dtype](
                self.intercept_scaling
            ),
            max_iter=self.max_iter,
            tol=Scalar[Self.compute_dtype](self.tol),
            random_state=self.random_state,
        )

        self.coef_ = res[0].copy()
        self.intercept_ = res[1]
        self.n_iter_ = res[2]
        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fit using a generic Dataset container.

        Args:
            dataset: Dataset containing feature matrix and target list.
        """
        self.fit(dataset.records, dataset.targets)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predict continuous target values for feature matrix X.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            List[Scalar[feat_dtype]]: Continuous predictions of length n_samples.

        Raises:
            NotFittedError: If model is not yet fitted.
            DimensionMismatchError: If feature dimension does not match n_features_in_.
        """
        check_is_fitted("LinearSVR", self.is_fitted)
        check_array[feat_dtype](X)

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "LinearSVR.predict",
            )

        var n_samples = X.rows
        var n_features = self.n_features_in_
        var preds = List[Scalar[feat_dtype]](capacity=n_samples)

        for i in range(n_samples):
            var val: Scalar[Self.compute_dtype] = 0.0
            for j in range(n_features):
                val += Scalar[Self.compute_dtype](X[i, j]) * self.coef_[j]
            if self.fit_intercept:
                val += self.intercept_
            preds.append(Scalar[feat_dtype](val))

        return preds^

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[
        Scalar[feat_dtype]
    ]:
        """Predict on features provided by a Dataset container.

        Args:
            dataset: Dataset container with records matrix.

        Returns:
            List[Scalar[feat_dtype]]: Continuous prediction vector.
        """
        return self.predict(dataset.records)

    def serialize(self, mut writer: BufferWriter):
        """Serializes LinearSVR parameters and fitted state into BufferWriter.
        """
        write_header(writer, "LinearSVR")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.epsilon)
        writer.write_float64(self.tol)
        writer.write_float64(self.C)
        writer.write_string(self.loss)
        writer.write_bool(self.fit_intercept)
        writer.write_float64(self.intercept_scaling)
        writer.write_bool(self.dual)
        writer.write_int(self.max_iter)
        writer.write_int(self.random_state)
        writer.write_int(self.n_features_in_)
        writer.write_int(self.n_iter_)
        writer.write_float_list[Self.compute_dtype](self.coef_)
        writer.write_float64(Float64(self.intercept_))

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes LinearSVR from BufferReader."""
        check_header(reader, "LinearSVR")
        var is_fitted = reader.read_bool()
        var epsilon = reader.read_float64()
        var tol = reader.read_float64()
        var C = reader.read_float64()
        var loss = reader.read_string()
        var fit_intercept = reader.read_bool()
        var intercept_scaling = reader.read_float64()
        var dual = reader.read_bool()
        var max_iter = reader.read_int()
        var random_state = reader.read_int()
        var n_features_in_ = reader.read_int()
        var n_iter_ = reader.read_int()
        var coef = reader.read_float_list[Self.compute_dtype]()
        var intercept = Scalar[Self.compute_dtype](reader.read_float64())

        var model = Self(
            epsilon=epsilon,
            tol=tol,
            C=C,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=dual,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.n_iter_ = n_iter_
        model.coef_ = coef^
        model.intercept_ = intercept
        return model^
