from ..core.matrix import Matrix
from ..core.dataset import Dataset
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
from ._kernels import (
    compute_gamma,
    compute_gram_matrix,
    compute_kernel_matrix,
)
from ._smo import _smo_binary_svc, SMOResult


struct SVC[compute_dtype: DType = DType.float64](
    Classifier, Copyable, Movable, Serializable
):
    """C-Support Vector Classification using non-linear kernels and SMO.

    Fits maximum-margin non-linear decision boundaries using Sequential
    Minimal Optimization (SMO) across arbitrary kernel mappings (RBF,
    Polynomial, Sigmoid, and Linear), with One-vs-Rest (OvR) multiclass scaling.

    Parameters:
        compute_dtype: Floating point precision for kernel operations. Default DType.float64.

    Args:
        C: Regularization parameter. Strength is inversely proportional to C.
            Must be strictly positive. Default 1.0.
        kernel: Kernel routine ('rbf', 'linear', 'poly', 'sigmoid'). Default 'rbf'.
        degree: Degree for polynomial kernel function. Ignored by others. Default 3.
        gamma: Kernel scaling mode ('scale' or 'auto'). Default 'scale'.
        gamma_val: Explicit numeric gamma. When > 0, overrides gamma mode. Default -1.0.
        coef0: Independent term in poly and sigmoid kernels. Default 0.0.
        tol: Tolerance for stopping criterion. Default 1e-3.
        max_iter: Hard limit on SMO solver iterations. Default 10000.
        decision_function_shape: Multiclass strategy ('ovr'). Default 'ovr'.

    Attributes:
        classes_: Unique sorted class labels seen during fit.
        support_vectors_: Extracted support vectors of shape (n_support_vectors, n_features).
        support_indices_: Row indices of support vectors in original training set.
        dual_coef_: Dual coefficients alpha_i * y_i for support vectors. Shape
            (1, n_support_vectors) for binary, (n_classes, n_support_vectors) for multiclass.
        intercept_: Bias terms b of length 1 for binary, length n_classes for multiclass.
        n_features_in_: Number of features seen during fit.
        n_iter_: Maximum iterations reached across binary solver subproblems.
        gamma_: Evaluated kernel scale coefficient.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.svm import SVC
        from strata.core import Matrix

        var clf = SVC[DType.float64](C=1.0, kernel="rbf")
        clf.fit(X_train, y_train)
        var preds = clf.predict(X_test)
        ```
    """

    var C: Float64
    var kernel: String
    var degree: Int
    var gamma: String
    var gamma_val: Float64
    var coef0: Float64
    var tol: Float64
    var max_iter: Int
    var decision_function_shape: String

    var classes_: List[Int]
    var support_vectors_: Matrix[Self.compute_dtype]
    var support_indices_: List[Int]
    var dual_coef_: Matrix[Self.compute_dtype]
    var intercept_: List[Scalar[Self.compute_dtype]]
    var n_features_in_: Int
    var n_iter_: Int
    var gamma_: Scalar[Self.compute_dtype]
    var is_fitted: Bool

    def __init__(
        out self,
        C: Float64 = 1.0,
        kernel: String = "rbf",
        degree: Int = 3,
        gamma: String = "scale",
        gamma_val: Float64 = -1.0,
        coef0: Float64 = 0.0,
        tol: Float64 = 1e-3,
        max_iter: Int = 10000,
        decision_function_shape: String = "ovr",
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.gamma_val = gamma_val
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape

        self.classes_ = List[Int]()
        self.support_vectors_ = Matrix[Self.compute_dtype](0, 0, 0.0)
        self.support_indices_ = List[Int]()
        self.dual_coef_ = Matrix[Self.compute_dtype](0, 0, 0.0)
        self.intercept_ = List[Scalar[Self.compute_dtype]]()
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.gamma_ = Scalar[Self.compute_dtype](0.0)
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        self.C = copy.C
        self.kernel = copy.kernel
        self.degree = copy.degree
        self.gamma = copy.gamma
        self.gamma_val = copy.gamma_val
        self.coef0 = copy.coef0
        self.tol = copy.tol
        self.max_iter = copy.max_iter
        self.decision_function_shape = copy.decision_function_shape

        self.classes_ = copy.classes_.copy()
        self.support_vectors_ = copy.support_vectors_.copy()
        self.support_indices_ = copy.support_indices_.copy()
        self.dual_coef_ = copy.dual_coef_.copy()
        self.intercept_ = copy.intercept_.copy()
        self.n_features_in_ = copy.n_features_in_
        self.n_iter_ = copy.n_iter_
        self.gamma_ = copy.gamma_
        self.is_fitted = copy.is_fitted

    def __init__(out self, *, deinit move: Self):
        self.C = move.C
        self.kernel = move.kernel
        self.degree = move.degree
        self.gamma = move.gamma
        self.gamma_val = move.gamma_val
        self.coef0 = move.coef0
        self.tol = move.tol
        self.max_iter = move.max_iter
        self.decision_function_shape = move.decision_function_shape

        self.classes_ = move.classes_^
        self.support_vectors_ = move.support_vectors_^
        self.support_indices_ = move.support_indices_^
        self.dual_coef_ = move.dual_coef_^
        self.intercept_ = move.intercept_^
        self.n_features_in_ = move.n_features_in_
        self.n_iter_ = move.n_iter_
        self.gamma_ = move.gamma_
        self.is_fitted = move.is_fitted

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fits the kernel Support Vector Classifier on feature matrix X and labels y.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target class labels of length n_samples.

        Raises:
            InvalidParameterError: If hyperparameters are out of range or fewer than 2 classes.
            DimensionMismatchError: If sample counts do not match.
        """
        check_X_y[feat_dtype, target_dtype](X, y)
        check_floating_dtype[Self.compute_dtype]()

        if self.C <= 0:
            raise InvalidParameterError.error(
                "C", "C must be strictly positive"
            )
        if self.tol <= 0:
            raise InvalidParameterError.error(
                "tol", "tol must be strictly positive"
            )
        if self.degree < 1:
            raise InvalidParameterError.error("degree", "degree must be >= 1")
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

        var l = X.rows
        var d = X.cols
        self.n_features_in_ = d

        # Extract unique sorted class labels
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

        # Bubble sort unique class labels
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                if unique_classes[i] > unique_classes[j]:
                    var temp = unique_classes[i]
                    unique_classes[i] = unique_classes[j]
                    unique_classes[j] = temp

        var K_classes = len(unique_classes)
        if K_classes < 2:
            raise InvalidParameterError.error(
                "y",
                "SVC requires at least 2 distinct classes, found "
                + String(K_classes),
            )

        self.classes_ = unique_classes^
        var X_comp = X.cast[Self.compute_dtype]()

        # Resolve gamma
        var gamma_val_scalar = Scalar[Self.compute_dtype](self.gamma_val)
        self.gamma_ = compute_gamma(X_comp, self.gamma, gamma_val_scalar)

        var coef0_scalar = Scalar[Self.compute_dtype](self.coef0)
        var C_scalar = Scalar[Self.compute_dtype](self.C)
        var tol_scalar = Scalar[Self.compute_dtype](self.tol)

        # Compute full symmetric Gram matrix on training features
        var Gram = compute_gram_matrix(
            X_comp, self.kernel, self.gamma_, self.degree, coef0_scalar
        )

        if K_classes == 2:
            # Binary classification: classes_[0] -> -1.0, classes_[1] -> +1.0
            var y_bin = List[Scalar[Self.compute_dtype]](capacity=l)
            var pos_class = self.classes_[1]
            for i in range(l):
                if Int(y[i]) == pos_class:
                    y_bin.append(Scalar[Self.compute_dtype](1.0))
                else:
                    y_bin.append(-Scalar[Self.compute_dtype](1.0))

            var res = _smo_binary_svc(
                Gram, y_bin, C_scalar, tol_scalar, self.max_iter
            )

            var n_sv = len(res.support_indices)
            self.support_indices_ = res.support_indices.copy()
            self.support_vectors_ = Matrix[Self.compute_dtype](n_sv, d, 0.0)
            self.dual_coef_ = Matrix[Self.compute_dtype](1, n_sv, 0.0)

            for m in range(n_sv):
                var sv_idx = res.support_indices[m]
                for j in range(d):
                    self.support_vectors_[m, j] = X_comp[sv_idx, j]
                self.dual_coef_[0, m] = res.dual_coef[m]

            var intercept_list = List[Scalar[Self.compute_dtype]](capacity=1)
            intercept_list.append(res.intercept)
            self.intercept_ = intercept_list^
            self.n_iter_ = res.n_iter
        else:
            # Multiclass OvR: train K_classes binary models
            var all_dual_coefs = List[List[Scalar[Self.compute_dtype]]]()
            var all_intercepts = List[Scalar[Self.compute_dtype]](
                capacity=K_classes
            )
            var all_sv_indices = List[List[Int]]()
            var max_iters = 0

            # Indicator array for support vector union
            var in_sv_union = List[Bool](capacity=l)
            for _ in range(l):
                in_sv_union.append(False)

            for k in range(K_classes):
                var target_k = self.classes_[k]
                var y_bin = List[Scalar[Self.compute_dtype]](capacity=l)
                for i in range(l):
                    if Int(y[i]) == target_k:
                        y_bin.append(Scalar[Self.compute_dtype](1.0))
                    else:
                        y_bin.append(-Scalar[Self.compute_dtype](1.0))

                var res = _smo_binary_svc(
                    Gram, y_bin, C_scalar, tol_scalar, self.max_iter
                )
                if res.n_iter > max_iters:
                    max_iters = res.n_iter

                for idx in range(len(res.support_indices)):
                    in_sv_union[res.support_indices[idx]] = True

                all_dual_coefs.append(res.dual_coef.copy())
                all_intercepts.append(res.intercept)
                all_sv_indices.append(res.support_indices.copy())

            # Consolidate unique support vector union
            var union_sv = List[Int]()
            for i in range(l):
                if in_sv_union[i]:
                    union_sv.append(i)

            var total_sv = len(union_sv)
            self.support_indices_ = union_sv.copy()
            self.support_vectors_ = Matrix[Self.compute_dtype](total_sv, d, 0.0)
            self.dual_coef_ = Matrix[Self.compute_dtype](
                K_classes, total_sv, 0.0
            )

            # Map sample index to SV position in union matrix
            var sv_map = List[Int](capacity=l)
            for _ in range(l):
                sv_map.append(-1)
            for m in range(total_sv):
                var orig_idx = union_sv[m]
                sv_map[orig_idx] = m
                for j in range(d):
                    self.support_vectors_[m, j] = X_comp[orig_idx, j]

            # Populate per-class dual coefficients
            for k in range(K_classes):
                ref sv_k = all_sv_indices[k]
                ref coef_k = all_dual_coefs[k]
                for idx in range(len(sv_k)):
                    var orig_idx = sv_k[idx]
                    var col_idx = sv_map[orig_idx]
                    self.dual_coef_[k, col_idx] = coef_k[idx]

            self.intercept_ = all_intercepts^
            self.n_iter_ = max_iters

        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fits the classifier using a unified Dataset container."""
        self.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)

    def decision_function[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Predict confidence scores for samples in X.

        Evaluates the kernel decision function against fitted support vectors:
        $$
        f(x) = \\sum_{m=1}^{N_{sv}} \\alpha_m y_m K(x_m, x) + b
        $$

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            Matrix[compute_dtype]: Shape (n_samples, 1) for binary, or
            (n_samples, n_classes) for multiclass.
        """
        check_is_fitted("SVC", self.is_fitted)
        check_array[feat_dtype](X)

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "SVC.decision_function",
            )

        var N = X.rows
        var K_classes = len(self.classes_)
        var S = self.support_vectors_.rows
        var X_comp = X.cast[Self.compute_dtype]()

        # When no support vectors were retained, return constant bias
        if S == 0:
            if K_classes == 2:
                var scores = Matrix[Self.compute_dtype](
                    N, 1, self.intercept_[0]
                )
                return scores^
            else:
                var scores = Matrix[Self.compute_dtype](N, K_classes, 0.0)
                for i in range(N):
                    for k in range(K_classes):
                        scores[i, k] = self.intercept_[k]
                return scores^

        # Kernel evaluation between test samples and support vectors: (N x S)
        var coef0_scalar = Scalar[Self.compute_dtype](self.coef0)
        var K_eval = compute_kernel_matrix(
            X_comp,
            self.support_vectors_,
            self.kernel,
            self.gamma_,
            self.degree,
            coef0_scalar,
        )

        if K_classes == 2:
            var scores = Matrix[Self.compute_dtype](N, 1, 0.0)
            var b = self.intercept_[0]
            for i in range(N):
                var dot: Scalar[Self.compute_dtype] = 0.0
                for m in range(S):
                    dot += K_eval[i, m] * self.dual_coef_[0, m]
                scores[i, 0] = dot + b
            return scores^
        else:
            var scores = Matrix[Self.compute_dtype](N, K_classes, 0.0)
            for i in range(N):
                for k in range(K_classes):
                    var dot: Scalar[Self.compute_dtype] = 0.0
                    for m in range(S):
                        dot += K_eval[i, m] * self.dual_coef_[k, m]
                    scores[i, k] = dot + self.intercept_[k]
            return scores^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for samples in X.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            List[Int]: Predicted discrete class labels for each sample.
        """
        var scores = self.decision_function(X)
        var N = scores.rows
        var K_classes = len(self.classes_)
        var preds = List[Int](capacity=N)

        if K_classes == 2:
            var neg_class = self.classes_[0]
            var pos_class = self.classes_[1]
            for i in range(N):
                if scores[i, 0] >= Scalar[Self.compute_dtype](0.0):
                    preds.append(pos_class)
                else:
                    preds.append(neg_class)
        else:
            for i in range(N):
                var best_idx = 0
                var best_score = scores[i, 0]
                for k in range(1, K_classes):
                    if scores[i, k] > best_score:
                        best_score = scores[i, k]
                        best_idx = k
                preds.append(self.classes_[best_idx])

        return preds^

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[Int]:
        """Predicts class labels for a Dataset container."""
        return self.predict[feat_dtype](dataset.records)

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Probability estimates using calibrated decision scores.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            Matrix[feat_dtype]: Probability matrix of shape (n_samples, n_classes).
        """
        var scores = self.decision_function(X)
        var N = scores.rows
        var K_classes = len(self.classes_)
        var probs = Matrix[feat_dtype](N, K_classes, 0.0)

        if K_classes == 2:
            for i in range(N):
                var s = scores[i, 0].cast[feat_dtype]()
                var p1 = sigmoid[feat_dtype](s)
                probs[i, 0] = Scalar[feat_dtype](1.0) - p1
                probs[i, 1] = p1
        else:
            for i in range(N):
                var row_scores = List[Scalar[feat_dtype]](capacity=K_classes)
                for k in range(K_classes):
                    row_scores.append(scores[i, k].cast[feat_dtype]())
                var row_probs = softmax[feat_dtype](row_scores)
                for k in range(K_classes):
                    probs[i, k] = row_probs[k]

        return probs^

    def predict_proba[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Matrix[
        feat_dtype
    ]:
        """Predicts class probability distributions for a Dataset container."""
        return self.predict_proba[feat_dtype](dataset.records)

    def serialize(self, mut writer: BufferWriter):
        """Serializes SVC parameters and fitted state into BufferWriter."""
        write_header(writer, "SVC")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.C)
        writer.write_string(self.kernel)
        writer.write_int(self.degree)
        writer.write_string(self.gamma)
        writer.write_float64(self.gamma_val)
        writer.write_float64(self.coef0)
        writer.write_float64(self.tol)
        writer.write_int(self.max_iter)
        writer.write_string(self.decision_function_shape)
        writer.write_int(self.n_features_in_)
        writer.write_int(self.n_iter_)
        writer.write_float64(Float64(self.gamma_))
        writer.write_int_list(self.classes_)
        writer.write_int_list(self.support_indices_)
        writer.write_matrix[Self.compute_dtype](self.support_vectors_)
        writer.write_matrix[Self.compute_dtype](self.dual_coef_)
        writer.write_float_list[Self.compute_dtype](self.intercept_)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes SVC from BufferReader."""
        check_header(reader, "SVC")
        var is_fitted = reader.read_bool()
        var C = reader.read_float64()
        var kernel = reader.read_string()
        var degree = reader.read_int()
        var gamma = reader.read_string()
        var gamma_val = reader.read_float64()
        var coef0 = reader.read_float64()
        var tol = reader.read_float64()
        var max_iter = reader.read_int()
        var decision_function_shape = reader.read_string()
        var n_features_in_ = reader.read_int()
        var n_iter_ = reader.read_int()
        var gamma_ = Scalar[Self.compute_dtype](reader.read_float64())
        var classes_ = reader.read_int_list()
        var support_indices_ = reader.read_int_list()
        var support_vectors_ = reader.read_matrix[Self.compute_dtype]()
        var dual_coef_ = reader.read_matrix[Self.compute_dtype]()
        var intercept_ = reader.read_float_list[Self.compute_dtype]()

        var model = Self(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            gamma_val=gamma_val,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.n_iter_ = n_iter_
        model.gamma_ = gamma_
        model.classes_ = classes_^
        model.support_indices_ = support_indices_^
        model.support_vectors_ = support_vectors_^
        model.dual_coef_ = dual_coef_^
        model.intercept_ = intercept_^
        return model^
