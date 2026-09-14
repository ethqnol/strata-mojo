from ..core.matrix import Matrix
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.math import sigmoid, softmax
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)
from ._linear_svm_fast import _dual_coordinate_descent_svm


struct LinearSVC[compute_dtype: DType = DType.float64](
    Classifier, Copyable, Movable, Serializable
):
    """Linear Support Vector Classification.

    Fits a maximum-margin linear boundary using L2 regularization and
    dual coordinate descent (LIBLINEAR algorithm). Supports both hinge
    loss (L1-SVM) and squared hinge loss (L2-SVM), with One-vs-Rest (OvR)
    multiclass scaling.

    Parameters:
        compute_dtype: Precision for floating-point matrix operations. Default DType.float64.

    Args:
        C: Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive. Default 1.0.
        loss: Loss function. Specifies the loss function. 'hinge' is standard
            SVM hinge loss; 'squared_hinge' is the square of the hinge loss.
            Default 'squared_hinge'.
        penalty: Norm used in the penalization. Only 'l2' is supported in dual mode.
            Default 'l2'.
        dual: Select the algorithm to solve the dual or primal optimization problem.
            Default True.
        tol: Tolerance for stopping criterion. Default 1e-4.
        fit_intercept: Whether to calculate the intercept for this model. Default True.
        intercept_scaling: When fit_intercept is True, instance vector x becomes
            [x, intercept_scaling]. Default 1.0.
        max_iter: The maximum number of coordinate descent iterations to run. Default 1000.
        random_state: Seed for pseudo-random number generator permutation. Default 42.

    Attributes:
        classes_: Unique sorted class labels seen during fit.
        coef_: Weights assigned to the features. Shape (1, n_features) for binary,
            (n_classes, n_features) for multiclass.
        intercept_: Constants in decision function. Length 1 for binary, length
            n_classes for multiclass.
        n_features_in_: Number of features seen during fit.
        n_iter_: Number of iterations run until convergence.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.svm import LinearSVC
        from strata.core import Matrix

        var X = Matrix[DType.float64](4, 2)
        X[0, 0] = 0.0; X[0, 1] = 0.0
        X[1, 0] = 0.1; X[1, 1] = 0.1
        X[2, 0] = 5.0; X[2, 1] = 5.0
        X[3, 0] = 5.1; X[3, 1] = 5.1

        var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]
        var clf = LinearSVC(C=1.0)
        clf.fit(X, y)
        var preds = clf.predict(X)
        ```
    """

    var C: Float64
    var loss: String
    var penalty: String
    var dual: Bool
    var tol: Float64
    var fit_intercept: Bool
    var intercept_scaling: Float64
    var max_iter: Int
    var random_state: Int

    var classes_: List[Int]
    var coef_: Matrix[Self.compute_dtype]
    var intercept_: List[Scalar[Self.compute_dtype]]
    var n_features_in_: Int
    var n_iter_: Int
    var is_fitted: Bool

    def __init__(
        out self,
        C: Float64 = 1.0,
        loss: String = "squared_hinge",
        penalty: String = "l2",
        dual: Bool = True,
        tol: Float64 = 1e-4,
        fit_intercept: Bool = True,
        intercept_scaling: Float64 = 1.0,
        max_iter: Int = 1000,
        random_state: Int = 42,
    ) raises:
        """Initialize LinearSVC with hyperparameters."""
        check_floating_dtype[Self.compute_dtype, "LinearSVC"]()

        if C <= 0.0:
            raise InvalidParameterError.error(
                "C", "C must be > 0.0, got " + String(C)
            )
        if loss != "squared_hinge" and loss != "hinge":
            raise InvalidParameterError.error(
                "loss",
                "loss must be 'squared_hinge' or 'hinge', got '" + loss + "'",
            )
        if penalty != "l2":
            raise InvalidParameterError.error(
                "penalty",
                "only 'l2' penalty is supported, got '" + penalty + "'",
            )
        if tol <= 0.0:
            raise InvalidParameterError.error(
                "tol", "tol must be > 0.0, got " + String(tol)
            )
        if intercept_scaling <= 0.0:
            raise InvalidParameterError.error(
                "intercept_scaling",
                "intercept_scaling must be > 0.0, got "
                + String(intercept_scaling),
            )
        if max_iter < 1:
            raise InvalidParameterError.error(
                "max_iter", "max_iter must be >= 1, got " + String(max_iter)
            )

        self.C = C
        self.loss = loss
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.max_iter = max_iter
        self.random_state = random_state

        self.classes_ = List[Int]()
        self.coef_ = Matrix[Self.compute_dtype](0, 0)
        self.intercept_ = List[Scalar[Self.compute_dtype]]()
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        """Copy constructor."""
        self.C = copy.C
        self.loss = copy.loss
        self.penalty = copy.penalty
        self.dual = copy.dual
        self.tol = copy.tol
        self.fit_intercept = copy.fit_intercept
        self.intercept_scaling = copy.intercept_scaling
        self.max_iter = copy.max_iter
        self.random_state = copy.random_state

        self.classes_ = copy.classes_.copy()
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_.copy()
        self.n_features_in_ = copy.n_features_in_
        self.n_iter_ = copy.n_iter_
        self.is_fitted = copy.is_fitted

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit the linear support vector classification model according to the given training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target label list of length n_samples.
        """
        check_X_y(X, y)
        check_array[feat_dtype](X)

        var l = X.rows
        var d = X.cols
        self.n_features_in_ = d

        # Find unique sorted classes
        var unique_classes = List[Int]()
        for i in range(l):
            var label = Int(y[i])
            var found = False
            for j in range(len(unique_classes)):
                if unique_classes[j] == label:
                    found = True
                    break
            if not found:
                unique_classes.append(label)

        # Sort classes in ascending order
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                if unique_classes[i] > unique_classes[j]:
                    var temp = unique_classes[i]
                    unique_classes[i] = unique_classes[j]
                    unique_classes[j] = temp

        var K = len(unique_classes)
        if K < 2:
            raise InvalidParameterError.error(
                "y",
                "LinearSVC requires at least 2 distinct classes, found "
                + String(K),
            )

        self.classes_ = unique_classes^
        var X_comp = X.cast[Self.compute_dtype]()
        var C_scalar = Scalar[Self.compute_dtype](self.C)
        var tol_scalar = Scalar[Self.compute_dtype](self.tol)
        var scaling_scalar = Scalar[Self.compute_dtype](self.intercept_scaling)

        if K == 2:
            # Binary classification: y in {-1, +1}
            # Convention: classes_[0] -> -1.0, classes_[1] -> +1.0
            var y_bin = List[Scalar[Self.compute_dtype]](capacity=l)
            var pos_class = self.classes_[1]
            for i in range(l):
                if Int(y[i]) == pos_class:
                    y_bin.append(Scalar[Self.compute_dtype](1.0))
                else:
                    y_bin.append(-Scalar[Self.compute_dtype](1.0))

            var res = _dual_coordinate_descent_svm[Self.compute_dtype](
                X_comp,
                y_bin,
                C_scalar,
                self.loss,
                self.fit_intercept,
                scaling_scalar,
                self.max_iter,
                tol_scalar,
                self.random_state,
            )

            var w = res[0].copy()
            var b = res[1]
            self.n_iter_ = res[2]

            var coef = Matrix[Self.compute_dtype](1, d)
            for j in range(d):
                coef[0, j] = w[j]
            self.coef_ = coef^

            var intercept = List[Scalar[Self.compute_dtype]]()
            intercept.append(b)
            self.intercept_ = intercept^

        else:
            # Multiclass classification: One-vs-Rest (OvR)
            var coef = Matrix[Self.compute_dtype](K, d)
            var intercept = List[Scalar[Self.compute_dtype]](capacity=K)
            var max_iters = 0

            for k in range(K):
                var target_class = self.classes_[k]
                var y_ovr = List[Scalar[Self.compute_dtype]](capacity=l)
                for i in range(l):
                    if Int(y[i]) == target_class:
                        y_ovr.append(Scalar[Self.compute_dtype](1.0))
                    else:
                        y_ovr.append(-Scalar[Self.compute_dtype](1.0))

                var res = _dual_coordinate_descent_svm[Self.compute_dtype](
                    X_comp,
                    y_ovr,
                    C_scalar,
                    self.loss,
                    self.fit_intercept,
                    scaling_scalar,
                    self.max_iter,
                    tol_scalar,
                    self.random_state + k * 17,
                )

                var w_k = res[0].copy()
                var b_k = res[1]
                var iters_k = res[2]
                if iters_k > max_iters:
                    max_iters = iters_k

                for j in range(d):
                    coef[k, j] = w_k[j]
                intercept.append(b_k)

            self.coef_ = coef^
            self.intercept_ = intercept^
            self.n_iter_ = max_iters

        self.is_fitted = True

    def decision_function[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Predict confidence scores for samples in X.

        The confidence score for a sample is the signed distance of that sample
        to the separating hyperplane.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            Matrix[compute_dtype]: Shape (n_samples, 1) for binary classification,
            or (n_samples, n_classes) for multiclass classification.
        """
        check_is_fitted("LinearSVC", self.is_fitted)
        check_array[feat_dtype](X)

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "LinearSVC.decision_function",
            )

        var N = X.rows
        var d = self.n_features_in_
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()

        if K == 2:
            var scores = Matrix[Self.compute_dtype](N, 1, 0)
            var b = self.intercept_[0]
            for i in range(N):
                var dot: Scalar[Self.compute_dtype] = 0.0
                for j in range(d):
                    dot += X_comp[i, j] * self.coef_[0, j]
                scores[i, 0] = dot + b
            return scores^
        else:
            var scores = Matrix[Self.compute_dtype](N, K, 0)
            for i in range(N):
                for k in range(K):
                    var dot: Scalar[Self.compute_dtype] = 0.0
                    for j in range(d):
                        dot += X_comp[i, j] * self.coef_[k, j]
                    scores[i, k] = dot + self.intercept_[k]
            return scores^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for samples in X.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            List[Int]: Predicted class labels for each sample.
        """
        var scores = self.decision_function(X)
        var N = scores.rows
        var K = len(self.classes_)
        var preds = List[Int](capacity=N)

        if K == 2:
            for i in range(N):
                if scores[i, 0] > Scalar[Self.compute_dtype](0.0):
                    preds.append(self.classes_[1])
                else:
                    preds.append(self.classes_[0])
        else:
            for i in range(N):
                var best_k = 0
                var max_score = scores[i, 0]
                for k in range(1, K):
                    var s = scores[i, k]
                    if s > max_score:
                        max_score = s
                        best_k = k
                preds.append(self.classes_[best_k])

        return preds^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Probability estimates using calibrated decision scores.

        Uses sigmoid calibration for binary classification and softmax
        normalization across OvR decision boundaries for multiclass classification.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            Matrix[feat_dtype]: Probability matrix of shape (n_samples, n_classes).
        """
        var scores = self.decision_function(X)
        var N = scores.rows
        var K = len(self.classes_)
        var probs = Matrix[feat_dtype](N, K, 0)

        if K == 2:
            for i in range(N):
                var s = scores[i, 0].cast[feat_dtype]()
                var p1 = sigmoid[feat_dtype](s)
                probs[i, 0] = Scalar[feat_dtype](1.0) - p1
                probs[i, 1] = p1
        else:
            for i in range(N):
                var row_scores = List[Scalar[feat_dtype]](capacity=K)
                for k in range(K):
                    row_scores.append(scores[i, k].cast[feat_dtype]())
                var row_probs = softmax[feat_dtype](row_scores)
                for k in range(K):
                    probs[i, k] = row_probs[k]

        return probs^

    def serialize(self, mut writer: BufferWriter):
        """Serializes LinearSVC parameters and fitted state into BufferWriter.
        """
        write_header(writer, "LinearSVC")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.C)
        writer.write_string(self.loss)
        writer.write_string(self.penalty)
        writer.write_bool(self.dual)
        writer.write_float64(self.tol)
        writer.write_bool(self.fit_intercept)
        writer.write_float64(self.intercept_scaling)
        writer.write_int(self.max_iter)
        writer.write_int(self.random_state)
        writer.write_int(self.n_features_in_)
        writer.write_int(self.n_iter_)
        writer.write_int_list(self.classes_)
        writer.write_matrix[Self.compute_dtype](self.coef_)
        writer.write_float_list[Self.compute_dtype](self.intercept_)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes LinearSVC from BufferReader."""
        check_header(reader, "LinearSVC")
        var is_fitted = reader.read_bool()
        var C = reader.read_float64()
        var loss = reader.read_string()
        var penalty = reader.read_string()
        var dual = reader.read_bool()
        var tol = reader.read_float64()
        var fit_intercept = reader.read_bool()
        var intercept_scaling = reader.read_float64()
        var max_iter = reader.read_int()
        var random_state = reader.read_int()
        var n_features_in_ = reader.read_int()
        var n_iter_ = reader.read_int()
        var classes_ = reader.read_int_list()
        var coef_ = reader.read_matrix[Self.compute_dtype]()
        var intercept_ = reader.read_float_list[Self.compute_dtype]()

        var model = Self(
            C=C,
            loss=loss,
            penalty=penalty,
            dual=dual,
            tol=tol,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.n_iter_ = n_iter_
        model.classes_ = classes_^
        model.coef_ = coef_^
        model.intercept_ = intercept_^
        return model^
