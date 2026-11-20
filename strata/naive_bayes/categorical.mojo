from std.math import log, exp, isnan, isinf
from ..core.matrix import Matrix
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.math import log_sum_exp
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


def _extract_category[
    dtype: DType
](val: Scalar[dtype], row: Int, col: Int) raises -> Int:
    """Validate and convert a scalar entry into a non-negative integer category index.
    """
    comptime if dtype == DType.bool:
        return 1 if val else 0
    elif dtype.is_floating_point():
        var fval = Float64(val)
        if isnan(fval) or isinf(fval):
            raise InvalidParameterError.error(
                "X",
                "NaN or infinite value in data passed to CategoricalNB at ("
                + String(row)
                + ", "
                + String(col)
                + ")",
            )
        if fval < 0.0:
            raise InvalidParameterError.error(
                "X",
                "Negative values in data passed to CategoricalNB at ("
                + String(row)
                + ", "
                + String(col)
                + ")",
            )
        if fval > 100000000.0:
            raise InvalidParameterError.error(
                "X",
                "Category index exceeds maximum supported value (100000000) at"
                " ("
                + String(row)
                + ", "
                + String(col)
                + ")",
            )
        var cat = Int(fval)
        if Float64(cat) != fval:
            raise InvalidParameterError.error(
                "X",
                "CategoricalNB requires non-negative integer category codes,"
                " got "
                + String(fval)
                + " at ("
                + String(row)
                + ", "
                + String(col)
                + ")",
            )
        return cat
    else:
        if val < 0:
            raise InvalidParameterError.error(
                "X",
                "Negative values in data passed to CategoricalNB at ("
                + String(row)
                + ", "
                + String(col)
                + ")",
            )
        var uval = val.cast[DType.uint64]()
        if uval > 100000000:
            raise InvalidParameterError.error(
                "X",
                "Category index exceeds maximum supported value (100000000) at"
                " ("
                + String(row)
                + ", "
                + String(col)
                + ")",
            )
        return Int(uval)


struct CategoricalNB[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable, Serializable):
    """Categorical Naive Bayes classifier for categorically distributed features.

    Suitable for discrete classification with features representing integer-coded
    categories ($0, 1, \\dots, n_j - 1$). The smoothed probability of category $t$
    in feature $j$ given class $c$ is:

    $$
    P(x_j = t \\mid y = c) = \\frac{N_{c, j, t} + \\alpha}{N_c + \\alpha \\cdot n_j}
    $$

    where $n_j$ is the number of available categories for feature $j$.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        alpha: Additive (Laplace/Lidstone) smoothing parameter. Must be non-negative.
            Default 1.0.
        fit_prior: Whether to learn class prior probabilities or assume a uniform
            prior. Default True.
        class_prior: Prior probabilities of the classes. If specified, the priors
            are not adjusted according to the data. Default empty.
        min_categories: Minimum number of categories per feature. Empty uses
            observed maximum + 1 per feature; length 1 applies uniformly; length D
            specifies per-feature minimums. Default empty.

    Attributes:
        classes_: Unique sorted class labels observed during fit.
        class_count_: Number of training samples observed in each class.
        class_log_prior_: Smoothed empirical log-probability of each class.
        category_count_: Per-feature matrices of shape $(K, n_j)$ holding sample
            counts for each (class, category) pair.
        feature_log_prob_: Per-feature matrices of shape $(K, n_j)$ holding empirical
            log probability $\\ln P(x_j = t \\mid y = c)$.
        n_categories_: Number of categories $n_j$ used for each feature.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.naive_bayes import CategoricalNB
        from strata.core import Matrix

        var cnb = CategoricalNB[DType.float64](alpha=1.0)
        cnb.fit(X_train, y_train)
        var preds = cnb.predict(X_test)
        var probs = cnb.predict_proba(X_test)
        ```
    """

    var is_fitted: Bool
    var alpha: Scalar[Self.compute_dtype]
    var fit_prior: Bool
    var class_prior: List[Scalar[Self.compute_dtype]]
    var min_categories: List[Int]
    var classes_: List[Int]
    var class_count_: List[Int]
    var class_log_prior_: List[Scalar[Self.compute_dtype]]
    var category_count_: List[Matrix[Self.compute_dtype]]
    var feature_log_prob_: List[Matrix[Self.compute_dtype]]
    var n_categories_: List[Int]
    var n_features_in_: Int

    def __init__(
        out self,
        alpha: Scalar[Self.compute_dtype] = 1.0,
        fit_prior: Bool = True,
        class_prior: List[Scalar[Self.compute_dtype]] = List[
            Scalar[Self.compute_dtype]
        ](),
        min_categories: List[Int] = List[Int](),
    ) raises:
        """Initialize the CategoricalNB classifier.

        Args:
            alpha: Additive smoothing parameter (>= 0). Default 1.0.
            fit_prior: Whether to learn class prior probabilities. Default True.
            class_prior: Prior probabilities of the classes. Default empty.
            min_categories: Minimum number of categories per feature. Default empty.

        Raises:
            InvalidParameterError: If alpha is negative/non-finite or any
                min_categories entry is less than 1.
        """
        check_floating_dtype[Self.compute_dtype, "CategoricalNB"]()
        var alpha_f64 = Float64(alpha)
        if isnan(alpha_f64) or isinf(alpha_f64) or alpha_f64 < 0.0:
            raise InvalidParameterError.error(
                "alpha",
                "alpha must be finite and non-negative, got " + String(alpha),
            )
        for j in range(len(min_categories)):
            if min_categories[j] < 1:
                raise InvalidParameterError.error(
                    "min_categories",
                    "min_categories entries must be >= 1, got "
                    + String(min_categories[j])
                    + " at index "
                    + String(j),
                )

        self.is_fitted = False
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior.copy()
        self.min_categories = min_categories.copy()
        self.classes_ = List[Int]()
        self.class_count_ = List[Int]()
        self.class_log_prior_ = List[Scalar[Self.compute_dtype]]()
        self.category_count_ = List[Matrix[Self.compute_dtype]]()
        self.feature_log_prob_ = List[Matrix[Self.compute_dtype]]()
        self.n_categories_ = List[Int]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copy an existing CategoricalNB instance."""
        self.is_fitted = copy.is_fitted
        self.alpha = copy.alpha
        self.fit_prior = copy.fit_prior
        self.class_prior = copy.class_prior.copy()
        self.min_categories = copy.min_categories.copy()
        self.classes_ = copy.classes_.copy()
        self.class_count_ = copy.class_count_.copy()
        self.class_log_prior_ = copy.class_log_prior_.copy()

        var cat_cnt = List[Matrix[Self.compute_dtype]](
            capacity=len(copy.category_count_)
        )
        for j in range(len(copy.category_count_)):
            cat_cnt.append(copy.category_count_[j].copy())
        self.category_count_ = cat_cnt^

        var feat_lp = List[Matrix[Self.compute_dtype]](
            capacity=len(copy.feature_log_prob_)
        )
        for j in range(len(copy.feature_log_prob_)):
            feat_lp.append(copy.feature_log_prob_[j].copy())
        self.feature_log_prob_ = feat_lp^

        self.n_categories_ = copy.n_categories_.copy()
        self.n_features_in_ = copy.n_features_in_

    def __init__(out self, *, deinit move: Self):
        """Move an existing CategoricalNB instance."""
        self.is_fitted = move.is_fitted
        self.alpha = move.alpha
        self.fit_prior = move.fit_prior
        self.class_prior = move.class_prior^
        self.min_categories = move.min_categories^
        self.classes_ = move.classes_^
        self.class_count_ = move.class_count_^
        self.class_log_prior_ = move.class_log_prior_^
        self.category_count_ = move.category_count_^
        self.feature_log_prob_ = move.feature_log_prob_^
        self.n_categories_ = move.n_categories_^
        self.n_features_in_ = move.n_features_in_

    def copy(self) -> Self:
        """Return a deep copy of the CategoricalNB instance."""
        return Self(copy=self)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit Categorical Naive Bayes using feature matrix X and target labels y.

        Args:
            X: Non-negative integer-coded categorical matrix of shape $(N, D)$.
            y: Target class labels of length $N$.

        Raises:
            InvalidParameterError: If X has negative, fractional, or non-finite
                values, classes < 2, or invalid priors/min_categories.
            DimensionMismatchError: If X is empty or X.rows != len(y).
        """
        check_X_y(X, y)

        var alpha_f64 = Float64(self.alpha)
        if isnan(alpha_f64) or isinf(alpha_f64) or alpha_f64 < 0.0:
            raise InvalidParameterError.error(
                "alpha",
                "alpha must be finite and non-negative, got "
                + String(self.alpha),
            )

        var N = X.rows
        var D = X.cols

        var min_cat_len = len(self.min_categories)
        if min_cat_len != 0 and min_cat_len != 1 and min_cat_len != D:
            raise InvalidParameterError.error(
                "min_categories",
                "min_categories length ("
                + String(min_cat_len)
                + ") must be 0, 1, or match n_features ("
                + String(D)
                + ")",
            )
        for j in range(min_cat_len):
            if self.min_categories[j] < 1:
                raise InvalidParameterError.error(
                    "min_categories",
                    "min_categories entries must be >= 1, got "
                    + String(self.min_categories[j]),
                )

        # Validate category values and determine maximum category index per feature
        var max_cats = List[Int](capacity=D)
        for _ in range(D):
            max_cats.append(0)

        var X_codes = List[Int](capacity=N * D)
        for i in range(N):
            for j in range(D):
                var code = _extract_category(X[i, j], i, j)
                X_codes.append(code)
                if code > max_cats[j]:
                    max_cats[j] = code

        var n_categories = List[Int](capacity=D)
        for j in range(D):
            var n_cat = max_cats[j] + 1
            if min_cat_len == 1:
                if self.min_categories[0] > n_cat:
                    n_cat = self.min_categories[0]
            elif min_cat_len == D:
                if self.min_categories[j] > n_cat:
                    n_cat = self.min_categories[j]
            n_categories.append(n_cat)

        # Extract unique classes and sort ascending
        var raw_classes = List[Int]()
        for i in range(N):
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
                "CategoricalNB requires at least 2 distinct classes, got "
                + String(K),
            )

        # Map each sample target to class index [0, K)
        var y_idx = List[Int](capacity=N)
        var class_counts = List[Int](capacity=K)
        for _ in range(K):
            class_counts.append(0)

        for i in range(N):
            var target_val = Int(y[i])
            var matched_k = 0
            for k in range(K):
                if raw_classes[k] == target_val:
                    matched_k = k
                    break
            y_idx.append(matched_k)
            class_counts[matched_k] += 1

        # Compute and validate class log priors
        var class_log_prior = List[Scalar[Self.compute_dtype]](capacity=K)
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
                if isnan(p) or isinf(p) or p <= 0.0:
                    raise InvalidParameterError.error(
                        "class_prior",
                        "Class prior must be finite and strictly positive.",
                    )
                p_sum += p
            var diff_from_one = p_sum - 1.0 if p_sum >= 1.0 else 1.0 - p_sum
            if diff_from_one > 1e-3:
                raise InvalidParameterError.error(
                    "class_prior",
                    "Sum of class priors must equal 1.0, got " + String(p_sum),
                )
            for k in range(K):
                class_log_prior.append(
                    Scalar[Self.compute_dtype](
                        log(Float64(self.class_prior[k]))
                    )
                )
        elif self.fit_prior:
            var log_N = log(Float64(N))
            for k in range(K):
                class_log_prior.append(
                    Scalar[Self.compute_dtype](
                        log(Float64(class_counts[k])) - log_N
                    )
                )
        else:
            var log_uniform = -log(Float64(K))
            for _ in range(K):
                class_log_prior.append(Scalar[Self.compute_dtype](log_uniform))

        # Accumulate category counts per feature and class
        var category_count = List[Matrix[Self.compute_dtype]](capacity=D)
        for j in range(D):
            category_count.append(
                Matrix[Self.compute_dtype](K, n_categories[j], 0)
            )

        for i in range(N):
            var k = y_idx[i]
            var row_offset = i * D
            for j in range(D):
                var cat = X_codes[row_offset + j]
                var cur = category_count[j][k, cat]
                category_count[j][k, cat] = cur + 1

        # Compute smoothed log probabilities per feature
        var feature_log_prob = List[Matrix[Self.compute_dtype]](capacity=D)
        for j in range(D):
            var n_cat = n_categories[j]
            var lp_mat = Matrix[Self.compute_dtype](K, n_cat, 0)
            for k in range(K):
                var smoothed_denom: Float64 = 0.0
                for t in range(n_cat):
                    smoothed_denom += (
                        Float64(category_count[j][k, t]) + alpha_f64
                    )

                var log_denom = (
                    log(smoothed_denom) if smoothed_denom
                    > 1e-15 else -34.538776
                )

                for t in range(n_cat):
                    var smoothed_num = (
                        Float64(category_count[j][k, t]) + alpha_f64
                    )
                    if smoothed_num > 1e-15:
                        lp_mat[k, t] = Scalar[Self.compute_dtype](
                            log(smoothed_num) - log_denom
                        )
                    else:
                        lp_mat[k, t] = Scalar[Self.compute_dtype](-1e9)

            feature_log_prob.append(lp_mat^)

        # Commit validated state atomically
        self.classes_ = raw_classes^
        self.class_count_ = class_counts^
        self.class_log_prior_ = class_log_prior^
        self.n_categories_ = n_categories^
        self.category_count_ = category_count^
        self.feature_log_prob_ = feature_log_prob^
        self.n_features_in_ = D
        self.is_fitted = True

    def _joint_log_likelihood[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Compute unnormalized joint log-likelihoods for samples in X."""
        check_is_fitted("CategoricalNB", self.is_fitted)
        check_array[feat_dtype](X)
        var D = self.n_features_in_
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "CategoricalNB._joint_log_likelihood",
            )

        var M = X.rows
        var K = len(self.classes_)
        var joint_ll = Matrix[Self.compute_dtype](M, K, 0)
        var row_acc = List[Float64](capacity=K)
        for _ in range(K):
            row_acc.append(0.0)

        for i in range(M):
            for k in range(K):
                row_acc[k] = Float64(self.class_log_prior_[k])

            for j in range(D):
                var cat = _extract_category(X[i, j], i, j)
                var n_cat = self.n_categories_[j]
                if cat >= n_cat:
                    raise InvalidParameterError.error(
                        "X",
                        "Category index "
                        + String(cat)
                        + " for feature "
                        + String(j)
                        + " is out of range [0, "
                        + String(n_cat)
                        + "). Use min_categories to reserve unseen categories.",
                    )
                for k in range(K):
                    row_acc[k] += Float64(self.feature_log_prob_[j][k, cat])

            for k in range(K):
                joint_ll[i, k] = Scalar[Self.compute_dtype](row_acc[k])

        return joint_ll^

    def predict_log_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class log probabilities for samples in X.

        Args:
            X: Categorical feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Log-probability matrix of shape $(N, K)$.

        Raises:
            DataConversionError: If `feat_dtype` is not a floating-point type.
            NotFittedError: If estimator is not fitted.
            DimensionMismatchError: If feature count does not match fit.
            InvalidParameterError: If X contains invalid or out-of-range categories.
        """
        comptime if not feat_dtype.is_floating_point():
            raise DataConversionError.error(
                "predict_log_proba requires a floating-point Matrix input to"
                " store log probabilities; cast X to Float64 or Float32 first."
            )

        var joint_ll = self._joint_log_likelihood(X)
        var M = joint_ll.rows
        var K = joint_ll.cols
        var log_probs = Matrix[feat_dtype](M, K, 0)

        for i in range(M):
            var row_vals = List[Scalar[Self.compute_dtype]](capacity=K)
            for k in range(K):
                row_vals.append(joint_ll[i, k])
            var lse = log_sum_exp[Self.compute_dtype](row_vals)
            for k in range(K):
                log_probs[i, k] = Scalar[feat_dtype](
                    Float64(joint_ll[i, k]) - Float64(lse)
                )

        return log_probs^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class probability distributions for samples in X.

        Args:
            X: Categorical feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Normalized probability matrix of shape $(N, K)$,
                where each row sums to 1.0.

        Raises:
            DataConversionError: If `feat_dtype` is not a floating-point type.
            NotFittedError: If estimator is not fitted.
            DimensionMismatchError: If feature count does not match fit.
            InvalidParameterError: If X contains invalid or out-of-range categories.
        """
        comptime if not feat_dtype.is_floating_point():
            raise DataConversionError.error(
                "predict_proba requires a floating-point Matrix input to store"
                " probabilities; cast X to Float64 or Float32 first."
            )

        var log_probs = self.predict_log_proba(X)
        var M = log_probs.rows
        var K = log_probs.cols
        var probs = Matrix[feat_dtype](M, K, 0)

        for i in range(M):
            var row_sum: Float64 = 0.0
            for k in range(K):
                var p = exp(Float64(log_probs[i, k]))
                row_sum += p
                probs[i, k] = Scalar[feat_dtype](p)
            if row_sum > 0.0:
                var inv_sum = 1.0 / row_sum
                for k in range(K):
                    probs[i, k] = Scalar[feat_dtype](
                        Float64(probs[i, k]) * inv_sum
                    )

        return probs^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for samples in X.

        Args:
            X: Categorical feature matrix of shape $(N, D)$.

        Returns:
            List[Int]: Predicted class labels of length $N$.

        Raises:
            NotFittedError: If estimator is not fitted.
            DimensionMismatchError: If feature count does not match fit.
            InvalidParameterError: If X contains invalid or out-of-range categories.
        """
        var joint_ll = self._joint_log_likelihood(X)
        var M = joint_ll.rows
        var K = joint_ll.cols
        var preds = List[Int](capacity=M)

        for i in range(M):
            var best_k = 0
            var max_val = joint_ll[i, 0]
            for k in range(1, K):
                if joint_ll[i, k] > max_val:
                    max_val = joint_ll[i, k]
                    best_k = k
            preds.append(self.classes_[best_k])

        return preds^

    def serialize(self, mut writer: BufferWriter):
        """Serialize CategoricalNB parameters and fitted state into BufferWriter.
        """
        write_header(writer, "CategoricalNB")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.alpha.cast[DType.float64]())
        writer.write_bool(self.fit_prior)
        writer.write_float_list[Self.compute_dtype](self.class_prior)
        writer.write_int_list(self.min_categories)
        writer.write_int(self.n_features_in_)
        writer.write_int_list(self.classes_)
        writer.write_int_list(self.class_count_)
        writer.write_float_list[Self.compute_dtype](self.class_log_prior_)
        writer.write_int_list(self.n_categories_)

        writer.write_int(len(self.category_count_))
        for j in range(len(self.category_count_)):
            writer.write_matrix[Self.compute_dtype](self.category_count_[j])

        writer.write_int(len(self.feature_log_prob_))
        for j in range(len(self.feature_log_prob_)):
            writer.write_matrix[Self.compute_dtype](self.feature_log_prob_[j])

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserialize CategoricalNB from BufferReader."""
        check_header(reader, "CategoricalNB")
        var is_fitted = reader.read_bool()
        var alpha = Scalar[Self.compute_dtype](reader.read_float64())
        var fit_prior = reader.read_bool()
        var class_prior = reader.read_float_list[Self.compute_dtype]()
        var min_categories = reader.read_int_list()
        var n_features_in_ = reader.read_int()
        if n_features_in_ < 0:
            raise DataConversionError.error(
                "Invalid serialized CategoricalNB: negative n_features_in_"
            )

        var classes_ = reader.read_int_list()
        var class_count_ = reader.read_int_list()
        var class_log_prior_ = reader.read_float_list[Self.compute_dtype]()
        var n_categories_ = reader.read_int_list()

        var num_cat_mats = reader.read_int()
        if num_cat_mats < 0:
            raise DataConversionError.error(
                "Invalid serialized CategoricalNB: negative category_count"
                " length"
            )
        var category_count_ = List[Matrix[Self.compute_dtype]](
            capacity=num_cat_mats
        )
        for _ in range(num_cat_mats):
            category_count_.append(reader.read_matrix[Self.compute_dtype]())

        var num_prob_mats = reader.read_int()
        if num_prob_mats < 0:
            raise DataConversionError.error(
                "Invalid serialized CategoricalNB: negative feature_log_prob"
                " length"
            )
        var feature_log_prob_ = List[Matrix[Self.compute_dtype]](
            capacity=num_prob_mats
        )
        for _ in range(num_prob_mats):
            feature_log_prob_.append(reader.read_matrix[Self.compute_dtype]())

        if is_fitted:
            var K = len(classes_)
            if (
                n_features_in_ < 1
                or num_cat_mats != n_features_in_
                or num_prob_mats != n_features_in_
                or len(n_categories_) != n_features_in_
                or K < 2
                or len(class_count_) != K
                or len(class_log_prior_) != K
            ):
                raise DataConversionError.error(
                    "Invalid serialized CategoricalNB: inconsistent fitted"
                    " state dimensions"
                )
            for j in range(n_features_in_):
                var n_cat = n_categories_[j]
                if (
                    n_cat < 1
                    or category_count_[j].rows != K
                    or category_count_[j].cols != n_cat
                    or feature_log_prob_[j].rows != K
                    or feature_log_prob_[j].cols != n_cat
                ):
                    raise DataConversionError.error(
                        "Invalid serialized CategoricalNB: feature matrix"
                        " shape mismatch"
                    )
        else:
            if (
                n_features_in_ != 0
                or num_cat_mats != 0
                or num_prob_mats != 0
                or len(classes_) != 0
                or len(class_count_) != 0
                or len(class_log_prior_) != 0
                or len(n_categories_) != 0
            ):
                raise DataConversionError.error(
                    "Invalid serialized CategoricalNB: unfitted model has"
                    " non-empty state"
                )

        var model = Self(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            min_categories=min_categories,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.classes_ = classes_^
        model.class_count_ = class_count_^
        model.class_log_prior_ = class_log_prior_^
        model.n_categories_ = n_categories_^
        model.category_count_ = category_count_^
        model.feature_log_prob_ = feature_log_prob_^
        return model^
