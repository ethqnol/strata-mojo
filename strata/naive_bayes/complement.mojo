from std.math import log, exp, abs
from ..core.matrix import Matrix
from ..core.csr_matrix import CSRMatrix
from ..core.dataset import Dataset
from ..core.linalg import gemm
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.math import log_sum_exp
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)


struct ComplementNB[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable, Serializable):
    """Complement Naive Bayes classifier (Rennie et al., 2003).

    Designed for text classification with imbalanced class distributions.
    Instead of calculating probabilities of features given class $c$,
    ComplementNB calculates probabilities of features given all classes *except* $c$,
    mitigating the majority class bias inherent in standard MultinomialNB.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        alpha: Additive (Laplace/Lidstone) smoothing parameter. Must be non-negative.
            Default 1.0.
        fit_prior: Whether to learn class prior probabilities or assume uniform.
            Default True.
        class_prior: Prior probabilities of the classes. If specified, the priors
            are not adjusted according to the data. Default empty.
        norm: Whether to perform a second normalization of the weights (Rennie et al. 2003).
            Default False.

    Attributes:
        classes_: Unique class labels observed during fit.
        class_count_: Number of training samples observed in each class.
        class_log_prior_: Smoothed empirical log-probability of each class.
        feature_count_: Number of samples encountered for each (class, feature)
            pair of shape $(K, D)$.
        feature_all_: Total count of each feature across all classes of length $D$.
        feature_log_prob_: Weights for decision making of shape $(K, D)$.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.naive_bayes import ComplementNB
        from strata.core import Matrix

        var cnb = ComplementNB[DType.float64](alpha=1.0)
        cnb.fit(X_train, y_train)
        var preds = cnb.predict(X_test)
        var probs = cnb.predict_proba(X_test)
        ```
    """

    var is_fitted: Bool
    var alpha: Scalar[Self.compute_dtype]
    var fit_prior: Bool
    var class_prior: List[Scalar[Self.compute_dtype]]
    var norm: Bool
    var classes_: List[Int]
    var class_count_: List[Int]
    var class_log_prior_: List[Scalar[Self.compute_dtype]]
    var feature_count_: Matrix[Self.compute_dtype]
    var feature_all_: List[Scalar[Self.compute_dtype]]
    var feature_log_prob_: Matrix[Self.compute_dtype]
    var n_features_in_: Int

    def __init__(
        out self,
        alpha: Scalar[Self.compute_dtype] = 1.0,
        fit_prior: Bool = True,
        class_prior: List[Scalar[Self.compute_dtype]] = List[
            Scalar[Self.compute_dtype]
        ](),
        norm: Bool = False,
    ) raises:
        """Initialize the ComplementNB classifier.

        Args:
            alpha: Additive smoothing parameter (>= 0). Default 1.0.
            fit_prior: Whether to learn class prior probabilities. Default True.
            class_prior: Prior probabilities of the classes. Default empty.
            norm: Whether to normalize weights. Default False.

        Raises:
            InvalidParameterError: If alpha is negative.
        """
        check_floating_dtype[Self.compute_dtype, "ComplementNB"]()
        if alpha < 0:
            raise InvalidParameterError.error(
                "alpha", "alpha must be non-negative, got " + String(alpha)
            )
        self.is_fitted = False
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior.copy()
        self.norm = norm
        self.classes_ = List[Int]()
        self.class_count_ = List[Int]()
        self.class_log_prior_ = List[Scalar[Self.compute_dtype]]()
        self.feature_count_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.feature_all_ = List[Scalar[Self.compute_dtype]]()
        self.feature_log_prob_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing ComplementNB instance."""
        self.is_fitted = copy.is_fitted
        self.alpha = copy.alpha
        self.fit_prior = copy.fit_prior
        self.class_prior = copy.class_prior.copy()
        self.norm = copy.norm
        self.classes_ = copy.classes_.copy()
        self.class_count_ = copy.class_count_.copy()
        self.class_log_prior_ = copy.class_log_prior_.copy()
        self.feature_count_ = copy.feature_count_.copy()
        self.feature_all_ = copy.feature_all_.copy()
        self.feature_log_prob_ = copy.feature_log_prob_.copy()
        self.n_features_in_ = copy.n_features_in_

    def copy(self) -> Self:
        """Returns a deep copy of the ComplementNB instance."""
        return Self(copy=self)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit Complement Naive Bayes using dense feature matrix X and target labels y.

        Args:
            X: Non-negative training feature matrix of shape $(N, D)$.
            y: Target class labels of length $N$.

        Raises:
            InvalidParameterError: If X contains negative values or classes < 2.
            DimensionMismatchError: If sample count of X does not match length of y.
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols
        self.n_features_in_ = D

        # Verify non-negativity
        for i in range(N):
            for j in range(D):
                if X[i, j] < 0:
                    raise InvalidParameterError.error(
                        "X",
                        "Negative values in data passed to ComplementNB at ("
                        + String(i)
                        + ", "
                        + String(j)
                        + ")",
                    )

        # Extract unique sorted classes
        var raw_classes = List[Int]()
        for i in range(len(y)):
            var label = Int(y[i])
            var found = False
            for c_idx in range(len(raw_classes)):
                if raw_classes[c_idx] == label:
                    found = True
                    break
            if not found:
                raw_classes.append(label)

        for i in range(len(raw_classes)):
            for j in range(i + 1, len(raw_classes)):
                if raw_classes[j] < raw_classes[i]:
                    var temp = raw_classes[i]
                    raw_classes[i] = raw_classes[j]
                    raw_classes[j] = temp

        var K = len(raw_classes)
        if K < 2:
            raise InvalidParameterError.error(
                "y",
                "ComplementNB requires at least 2 distinct classes, got "
                + String(K),
            )
        self.classes_ = raw_classes^

        var y_idx = List[Int](capacity=N)
        var class_counts = List[Int](capacity=K)
        for _ in range(K):
            class_counts.append(0)

        for i in range(N):
            var target_val = Int(y[i])
            var matched_k = 0
            for k in range(K):
                if self.classes_[k] == target_val:
                    matched_k = k
                    break
            y_idx.append(matched_k)
            class_counts[matched_k] += 1

        self.class_count_ = class_counts^

        # Feature counts per class
        self.feature_count_ = Matrix[Self.compute_dtype](K, D, 0)
        for i in range(N):
            var k = y_idx[i]
            for j in range(D):
                self.feature_count_[k, j] += Scalar[Self.compute_dtype](X[i, j])

        self._compute_weights(N, K, D)
        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: CSRMatrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit Complement Naive Bayes using sparse CSRMatrix and target labels y.

        Args:
            X: Non-negative sparse CSRMatrix of shape $(N, D)$.
            y: Target class labels of length $N$.

        Raises:
            InvalidParameterError: If X contains negative values or classes < 2.
            DimensionMismatchError: If sample count of X does not match length of y.
        """
        var N = X.rows
        var D = X.cols
        if N != len(y):
            raise DimensionMismatchError.error(
                "X.rows == len(y)",
                "X.rows=" + String(N) + ", len(y)=" + String(len(y)),
                "ComplementNB.fit",
            )
        if N == 0 or D == 0:
            raise InvalidParameterError.error(
                "X", "Input matrix cannot be empty."
            )

        self.n_features_in_ = D

        var raw_classes = List[Int]()
        for i in range(len(y)):
            var label = Int(y[i])
            var found = False
            for c_idx in range(len(raw_classes)):
                if raw_classes[c_idx] == label:
                    found = True
                    break
            if not found:
                raw_classes.append(label)

        for i in range(len(raw_classes)):
            for j in range(i + 1, len(raw_classes)):
                if raw_classes[j] < raw_classes[i]:
                    var temp = raw_classes[i]
                    raw_classes[i] = raw_classes[j]
                    raw_classes[j] = temp

        var K = len(raw_classes)
        if K < 2:
            raise InvalidParameterError.error(
                "y",
                "ComplementNB requires at least 2 distinct classes, got "
                + String(K),
            )
        self.classes_ = raw_classes^

        var y_idx = List[Int](capacity=N)
        var class_counts = List[Int](capacity=K)
        for _ in range(K):
            class_counts.append(0)

        for i in range(N):
            var target_val = Int(y[i])
            var matched_k = 0
            for k in range(K):
                if self.classes_[k] == target_val:
                    matched_k = k
                    break
            y_idx.append(matched_k)
            class_counts[matched_k] += 1

        self.class_count_ = class_counts^

        self.feature_count_ = Matrix[Self.compute_dtype](K, D, 0)
        for i in range(N):
            var k = y_idx[i]
            var start_idx = X.indptr[i]
            var end_idx = X.indptr[i + 1]
            for idx in range(start_idx, end_idx):
                var col = X.indices[idx]
                var val = Scalar[Self.compute_dtype](X.data[idx])
                if val < 0:
                    raise InvalidParameterError.error(
                        "X",
                        "Negative values in sparse data passed to ComplementNB",
                    )
                self.feature_count_[k, col] += val

        self._compute_weights(N, K, D)
        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fit using a generic Dataset container."""
        self.fit(dataset.records, dataset.targets)

    def _compute_weights(mut self, N: Int, K: Int, D: Int) raises:
        """Compute complement feature weights and class log priors."""
        # Total count per feature across all classes
        self.feature_all_ = List[Scalar[Self.compute_dtype]](capacity=D)
        for j in range(D):
            var sum_j: Scalar[Self.compute_dtype] = 0.0
            for k in range(K):
                sum_j += self.feature_count_[k, j]
            self.feature_all_.append(sum_j)

        self.feature_log_prob_ = Matrix[Self.compute_dtype](K, D, 0)

        # comp_count = feature_all_ + alpha - feature_count_
        for k in range(K):
            var comp_counts = List[Scalar[Self.compute_dtype]](capacity=D)
            var comp_sum: Scalar[Self.compute_dtype] = 0.0
            for j in range(D):
                var cc = (
                    self.feature_all_[j]
                    + self.alpha
                    - self.feature_count_[k, j]
                )
                comp_counts.append(cc)
                comp_sum += cc

            var log_denom = log(Float64(comp_sum))
            var logged_row = List[Float64](capacity=D)
            var summed_logged: Float64 = 0.0

            for j in range(D):
                var logged_val = log(Float64(comp_counts[j])) - log_denom
                logged_row.append(logged_val)
                summed_logged += logged_val

            if self.norm:
                for j in range(D):
                    var norm_val = logged_row[j] / summed_logged
                    self.feature_log_prob_[k, j] = Scalar[Self.compute_dtype](
                        norm_val
                    )
            else:
                for j in range(D):
                    self.feature_log_prob_[k, j] = Scalar[Self.compute_dtype](
                        -logged_row[j]
                    )

        # Class log priors
        self.class_log_prior_ = List[Scalar[Self.compute_dtype]](capacity=K)
        if len(self.class_prior) > 0:
            if len(self.class_prior) != K:
                raise InvalidParameterError.error(
                    "class_prior",
                    "Number of priors ("
                    + String(len(self.class_prior))
                    + ") must match number of classes ("
                    + String(K)
                    + ")",
                )
            var p_sum: Float64 = 0.0
            for k in range(K):
                var p = Float64(self.class_prior[k])
                if p <= 0:
                    raise InvalidParameterError.error(
                        "class_prior", "Class prior must be strictly positive."
                    )
                p_sum += p
            var diff_from_one = p_sum - 1.0 if p_sum >= 1.0 else 1.0 - p_sum
            if diff_from_one > 1e-3:
                raise InvalidParameterError.error(
                    "class_prior",
                    "Sum of class priors must equal 1.0, got " + String(p_sum),
                )
            for k in range(K):
                self.class_log_prior_.append(
                    Scalar[Self.compute_dtype](
                        log(Float64(self.class_prior[k]))
                    )
                )
        elif self.fit_prior:
            var log_N = log(Float64(N))
            for k in range(K):
                self.class_log_prior_.append(
                    Scalar[Self.compute_dtype](
                        log(Float64(self.class_count_[k])) - log_N
                    )
                )
        else:
            var log_uniform = -log(Float64(K))
            for _ in range(K):
                self.class_log_prior_.append(
                    Scalar[Self.compute_dtype](log_uniform)
                )

    def _joint_log_likelihood[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Calculates decision scores for samples in dense Matrix X."""
        check_is_fitted("ComplementNB", self.is_fitted)
        check_array[feat_dtype](X)
        var D = self.feature_log_prob_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "ComplementNB._joint_log_likelihood",
            )

        var M = X.rows
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()

        # GEMM: (M x D) @ (D x K) -> (M x K)
        var W_T = self.feature_log_prob_.transpose()
        var jll = gemm(X_comp, W_T)

        if K == 1:
            for i in range(M):
                jll[i, 0] += self.class_log_prior_[0]

        return jll^

    def _joint_log_likelihood[
        feat_dtype: DType
    ](self, X: CSRMatrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Calculates decision scores for samples in sparse CSRMatrix X."""
        check_is_fitted("ComplementNB", self.is_fitted)
        var D = self.feature_log_prob_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "ComplementNB._joint_log_likelihood",
            )

        var M = X.rows
        var K = len(self.classes_)

        var jll = Matrix[Self.compute_dtype](M, K, 0)
        for i in range(M):
            var start_idx = X.indptr[i]
            var end_idx = X.indptr[i + 1]
            for idx in range(start_idx, end_idx):
                var col = X.indices[idx]
                var val = Scalar[Self.compute_dtype](X.data[idx])
                for k in range(K):
                    jll[i, k] += val * self.feature_log_prob_[k, col]

            if K == 1:
                jll[i, 0] += self.class_log_prior_[0]

        return jll^

    def predict_log_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class log probabilities for dense matrix X."""
        var jll = self._joint_log_likelihood(X)
        var M = jll.rows
        var K = jll.cols

        var log_prob = Matrix[feat_dtype](M, K, 0)
        for i in range(M):
            var row_vals = List[Scalar[Self.compute_dtype]](capacity=K)
            for k in range(K):
                row_vals.append(jll[i, k])
            var lse = log_sum_exp(row_vals)
            for k in range(K):
                log_prob[i, k] = Scalar[feat_dtype](jll[i, k] - lse)

        return log_prob^

    def predict_log_proba[
        feat_dtype: DType
    ](self, X: CSRMatrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class log probabilities for sparse CSRMatrix X."""
        var jll = self._joint_log_likelihood(X)
        var M = jll.rows
        var K = jll.cols

        var log_prob = Matrix[feat_dtype](M, K, 0)
        for i in range(M):
            var row_vals = List[Scalar[Self.compute_dtype]](capacity=K)
            for k in range(K):
                row_vals.append(jll[i, k])
            var lse = log_sum_exp(row_vals)
            for k in range(K):
                log_prob[i, k] = Scalar[feat_dtype](jll[i, k] - lse)

        return log_prob^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class probabilities for dense matrix X."""
        var log_prob = self.predict_log_proba(X)
        var M = log_prob.rows
        var K = log_prob.cols
        var prob = Matrix[feat_dtype](M, K, 0)
        for i in range(M):
            for k in range(K):
                prob[i, k] = Scalar[feat_dtype](exp(Float64(log_prob[i, k])))
        return prob^

    def predict_proba[
        feat_dtype: DType
    ](self, X: CSRMatrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class probabilities for sparse CSRMatrix X."""
        var log_prob = self.predict_log_proba(X)
        var M = log_prob.rows
        var K = log_prob.cols
        var prob = Matrix[feat_dtype](M, K, 0)
        for i in range(M):
            for k in range(K):
                prob[i, k] = Scalar[feat_dtype](exp(Float64(log_prob[i, k])))
        return prob^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for samples in dense Matrix X."""
        var jll = self._joint_log_likelihood(X)
        var M = jll.rows
        var K = jll.cols
        var preds = List[Int](capacity=M)
        for i in range(M):
            var best_idx = 0
            var best_val = jll[i, 0]
            for k in range(1, K):
                if jll[i, k] > best_val:
                    best_val = jll[i, k]
                    best_idx = k
            preds.append(self.classes_[best_idx])
        return preds^

    def predict[
        feat_dtype: DType
    ](self, X: CSRMatrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for samples in sparse CSRMatrix X."""
        var jll = self._joint_log_likelihood(X)
        var M = jll.rows
        var K = jll.cols
        var preds = List[Int](capacity=M)
        for i in range(M):
            var best_idx = 0
            var best_val = jll[i, 0]
            for k in range(1, K):
                if jll[i, k] > best_val:
                    best_val = jll[i, k]
                    best_idx = k
            preds.append(self.classes_[best_idx])
        return preds^

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[Int]:
        """Predict class labels on a generic Dataset container."""
        return self.predict(dataset.records)

    def predict_proba[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Matrix[
        feat_dtype
    ]:
        """Predict class probabilities on a generic Dataset container."""
        return self.predict_proba(dataset.records)

    def serialize(self, mut writer: BufferWriter):
        """Serializes ComplementNB parameters and fitted state into BufferWriter.
        """
        write_header(writer, "ComplementNB")
        writer.write_bool(self.is_fitted)
        writer.write_float64(Float64(self.alpha))
        writer.write_bool(self.fit_prior)
        writer.write_bool(self.norm)
        writer.write_int(self.n_features_in_)
        writer.write_float_list[Self.compute_dtype](self.class_prior)
        writer.write_int_list(self.classes_)
        writer.write_int_list(self.class_count_)
        writer.write_float_list[Self.compute_dtype](self.class_log_prior_)
        writer.write_matrix[Self.compute_dtype](self.feature_count_)
        writer.write_float_list[Self.compute_dtype](self.feature_all_)
        writer.write_matrix[Self.compute_dtype](self.feature_log_prob_)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes ComplementNB from BufferReader."""
        check_header(reader, "ComplementNB")
        var is_fitted = reader.read_bool()
        var alpha = Scalar[Self.compute_dtype](reader.read_float64())
        var fit_prior = reader.read_bool()
        var norm = reader.read_bool()
        var n_features_in_ = reader.read_int()
        var class_prior = reader.read_float_list[Self.compute_dtype]()
        var classes_ = reader.read_int_list()
        var class_count_ = reader.read_int_list()
        var class_log_prior_ = reader.read_float_list[Self.compute_dtype]()
        var feature_count_ = reader.read_matrix[Self.compute_dtype]()
        var feature_all_ = reader.read_float_list[Self.compute_dtype]()
        var feature_log_prob_ = reader.read_matrix[Self.compute_dtype]()

        var model = Self(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior^,
            norm=norm,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.classes_ = classes_^
        model.class_count_ = class_count_^
        model.class_log_prior_ = class_log_prior_^
        model.feature_count_ = feature_count_^
        model.feature_all_ = feature_all_^
        model.feature_log_prob_ = feature_log_prob_^
        return model^
