from std.math import nan, inf, abs
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    assert_almost_equal,
)
from strata.core.matrix import Matrix
from strata.core.dataset import Dataset
from strata.base.estimator import (
    fit as fit_dataset,
    predict as predict_dataset,
)
from strata.svm.linear_svc import LinearSVC
from strata.io.serializer import dumps, loads
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_linear_svc_linearly_separable_binary() raises:
    # 4 points for class 0, 4 points for class 1
    var X = Matrix[DType.float64](8, 2, 0)
    # Class 0: around (-2, -2)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -2.1
    X[1, 1] = -1.9
    X[2, 0] = -1.9
    X[2, 1] = -2.1
    X[3, 0] = -1.8
    X[3, 1] = -1.8

    # Class 1: around (2, 2)
    X[4, 0] = 2.0
    X[4, 1] = 2.0
    X[5, 0] = 2.1
    X[5, 1] = 1.9
    X[6, 0] = 1.9
    X[6, 1] = 2.1
    X[7, 0] = 1.8
    X[7, 1] = 1.8

    var y = List[Scalar[DType.int32]](capacity=8)
    for _ in range(4):
        y.append(0)
    for _ in range(4):
        y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0, loss="squared_hinge")
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(len(clf.classes_), 2)
    assert_equal(clf.classes_[0], 0)
    assert_equal(clf.classes_[1], 1)
    assert_equal(clf.coef_.rows, 1)
    assert_equal(clf.coef_.cols, 2)
    assert_equal(len(clf.intercept_), 1)

    var preds = clf.predict(X)
    assert_equal(len(preds), 8)
    for i in range(4):
        assert_equal(preds[i], 0)
    for i in range(4, 8):
        assert_equal(preds[i], 1)


def test_linear_svc_hinge_loss() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -3.0
    X[0, 1] = -3.0
    X[1, 0] = -2.0
    X[1, 1] = -2.5
    X[2, 0] = -2.5
    X[2, 1] = -2.0
    X[3, 0] = 3.0
    X[3, 1] = 3.0
    X[4, 0] = 2.0
    X[4, 1] = 2.5
    X[5, 0] = 2.5
    X[5, 1] = 2.0

    var y = List[Scalar[DType.int32]](capacity=6)
    for _ in range(3):
        y.append(0)
    for _ in range(3):
        y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0, loss="hinge")
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    var preds = clf.predict(X)
    for i in range(3):
        assert_equal(preds[i], 0)
    for i in range(3, 6):
        assert_equal(preds[i], 1)


def test_linear_svc_decision_function_binary() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -5.0
    X[0, 1] = -5.0
    X[1, 0] = -4.0
    X[1, 1] = -4.0
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 4.0
    X[3, 1] = 4.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    var scores = clf.decision_function(X)
    assert_equal(scores.rows, 4)
    assert_equal(scores.cols, 1)

    # Class 0 points should have negative scores, class 1 positive scores
    assert_true(Float64(scores[0, 0]) < 0.0)
    assert_true(Float64(scores[1, 0]) < 0.0)
    assert_true(Float64(scores[2, 0]) > 0.0)
    assert_true(Float64(scores[3, 0]) > 0.0)


def test_linear_svc_predict_proba_binary() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -5.0
    X[0, 1] = -5.0
    X[1, 0] = -4.0
    X[1, 1] = -4.0
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 4.0
    X[3, 1] = 4.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 4)
    assert_equal(probs.cols, 2)

    for i in range(4):
        # Probabilities sum to 1.0
        var p_sum = Float64(probs[i, 0]) + Float64(probs[i, 1])
        assert_almost_equal(p_sum, 1.0, atol=1e-6)

    # Class 0 points should have P(0) > P(1)
    assert_true(Float64(probs[0, 0]) > Float64(probs[0, 1]))
    assert_true(Float64(probs[1, 0]) > Float64(probs[1, 1]))
    # Class 1 points should have P(1) > P(0)
    assert_true(Float64(probs[2, 1]) > Float64(probs[2, 0]))
    assert_true(Float64(probs[3, 1]) > Float64(probs[3, 0]))


def test_linear_svc_multiclass_ovr_3class() raises:
    # 3 classes: Class 0 at (0, 10), Class 1 at (-10, -10), Class 2 at (10, -10)
    var X = Matrix[DType.float64](9, 2, 0)
    # Class 0
    X[0, 0] = 0.0
    X[0, 1] = 10.0
    X[1, 0] = 0.1
    X[1, 1] = 9.9
    X[2, 0] = -0.1
    X[2, 1] = 10.1

    # Class 1
    X[3, 0] = -10.0
    X[3, 1] = -10.0
    X[4, 0] = -10.1
    X[4, 1] = -9.9
    X[5, 0] = -9.9
    X[5, 1] = -10.1

    # Class 2
    X[6, 0] = 10.0
    X[6, 1] = -10.0
    X[7, 0] = 10.1
    X[7, 1] = -9.9
    X[8, 0] = 9.9
    X[8, 1] = -10.1

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)
    y.append(1)
    y.append(2)
    y.append(2)
    y.append(2)

    var clf = LinearSVC[DType.float64](C=1.0, max_iter=2000)
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 3)
    assert_equal(clf.coef_.rows, 3)
    assert_equal(clf.coef_.cols, 2)
    assert_equal(len(clf.intercept_), 3)

    var preds = clf.predict(X)
    assert_equal(len(preds), 9)
    for i in range(3):
        assert_equal(preds[i], 0)
    for i in range(3, 6):
        assert_equal(preds[i], 1)
    for i in range(6, 9):
        assert_equal(preds[i], 2)

    # Test decision function shape
    var scores = clf.decision_function(X)
    assert_equal(scores.rows, 9)
    assert_equal(scores.cols, 3)

    # Test predict_proba shape and row sums
    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 9)
    assert_equal(probs.cols, 3)
    for i in range(9):
        var p_sum = (
            Float64(probs[i, 0]) + Float64(probs[i, 1]) + Float64(probs[i, 2])
        )
        assert_almost_equal(p_sum, 1.0, atol=1e-5)


def test_linear_svc_no_intercept() raises:
    # Points through origin
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -1.0
    X[1, 0] = -3.0
    X[1, 1] = -2.0
    X[2, 0] = 2.0
    X[2, 1] = 1.0
    X[3, 0] = 3.0
    X[3, 1] = 2.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0, fit_intercept=False)
    clf.fit(X, y)

    assert_equal(len(clf.intercept_), 1)
    assert_almost_equal(Float64(clf.intercept_[0]), 0.0, atol=1e-12)

    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_linear_svc_intercept_scaling() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -1.0
    X[0, 1] = -1.0
    X[1, 0] = -2.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[3, 0] = 2.0
    X[3, 1] = 2.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var clf1 = LinearSVC[DType.float64](intercept_scaling=0.5)
    clf1.fit(X, y)

    var clf2 = LinearSVC[DType.float64](intercept_scaling=2.0)
    clf2.fit(X, y)

    assert_true(clf1.is_fitted)
    assert_true(clf2.is_fitted)
    var p1 = clf1.predict(X)
    var p2 = clf2.predict(X)
    for i in range(4):
        assert_equal(p1[i], p2[i])


def test_linear_svc_c_regularization_sweep() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -3.0
    X[0, 1] = -3.0
    X[1, 0] = -2.0
    X[1, 1] = -2.0
    X[2, 0] = -1.0
    X[2, 1] = -1.0
    X[3, 0] = 1.0
    X[3, 1] = 1.0
    X[4, 0] = 2.0
    X[4, 1] = 2.0
    X[5, 0] = 3.0
    X[5, 1] = 3.0

    var y = List[Scalar[DType.int32]]()
    for _ in range(3):
        y.append(0)
    for _ in range(3):
        y.append(1)

    # Strong regularization (C=0.01) -> smaller weights
    var clf_small_c = LinearSVC[DType.float64](C=0.01)
    clf_small_c.fit(X, y)
    var norm_small = abs(Float64(clf_small_c.coef_[0, 0])) + abs(
        Float64(clf_small_c.coef_[0, 1])
    )

    # Weak regularization (C=100.0) -> larger weights
    var clf_large_c = LinearSVC[DType.float64](C=100.0)
    clf_large_c.fit(X, y)
    var norm_large = abs(Float64(clf_large_c.coef_[0, 0])) + abs(
        Float64(clf_large_c.coef_[0, 1])
    )

    assert_true(norm_large > norm_small)


def test_linear_svc_high_dimensional() raises:
    var n = 10
    var d = 15
    var X = Matrix[DType.float64](n, d, 0)
    var y = List[Scalar[DType.int32]](capacity=n)

    for i in range(n // 2):
        for j in range(d):
            X[i, j] = -1.0 - Float64(j) * 0.1
        y.append(0)

    for i in range(n // 2, n):
        for j in range(d):
            X[i, j] = 1.0 + Float64(j) * 0.1
        y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    assert_equal(clf.n_features_in_, d)
    assert_equal(clf.coef_.cols, d)
    var preds = clf.predict(X)
    for i in range(n // 2):
        assert_equal(preds[i], 0)
    for i in range(n // 2, n):
        assert_equal(preds[i], 1)


def test_linear_svc_float32() raises:
    var X = Matrix[DType.float32](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -1.0
    X[1, 1] = -1.0
    X[2, 0] = 2.0
    X[2, 1] = 2.0
    X[3, 0] = 1.0
    X[3, 1] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var clf = LinearSVC[DType.float32](C=1.0)
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_linear_svc_copy_semantics() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -1.0
    X[1, 1] = -1.0
    X[2, 0] = 2.0
    X[2, 1] = 2.0
    X[3, 0] = 1.0
    X[3, 1] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    var clf_copy = LinearSVC(copy=clf)
    assert_equal(clf_copy.n_features_in_, clf.n_features_in_)
    assert_equal(len(clf_copy.classes_), len(clf.classes_))
    assert_almost_equal(
        Float64(clf_copy.coef_[0, 0]), Float64(clf.coef_[0, 0]), atol=1e-12
    )
    assert_almost_equal(
        Float64(clf_copy.intercept_[0]),
        Float64(clf.intercept_[0]),
        atol=1e-12,
    )

    var preds_orig = clf.predict(X)
    var preds_copy = clf_copy.predict(X)
    for i in range(4):
        assert_equal(preds_orig[i], preds_copy[i])


def test_linear_svc_serialization_roundtrip() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -3.0
    X[0, 1] = -3.0
    X[1, 0] = -2.0
    X[1, 1] = -2.0
    X[2, 0] = 3.0
    X[2, 1] = 3.0
    X[3, 0] = 2.0
    X[3, 1] = 2.0

    var y = List[Scalar[DType.int32]]()
    y.append(10)
    y.append(10)
    y.append(20)
    y.append(20)

    var clf = LinearSVC[DType.float64](C=2.5, loss="squared_hinge")
    clf.fit(X, y)

    var serialized_data = dumps(clf)
    var restored = loads[LinearSVC[DType.float64]](serialized_data)

    assert_true(restored.is_fitted)
    assert_equal(restored.C, clf.C)
    assert_equal(restored.loss, clf.loss)
    assert_equal(restored.n_features_in_, clf.n_features_in_)
    assert_equal(restored.classes_[0], 10)
    assert_equal(restored.classes_[1], 20)

    var preds_orig = clf.predict(X)
    var preds_restored = restored.predict(X)
    for i in range(4):
        assert_equal(preds_orig[i], preds_restored[i])


def test_linear_svc_dataset_interop() raises:
    var records = Matrix[DType.float64](4, 2, 0)
    records[0, 0] = -1.0
    records[0, 1] = -1.0
    records[1, 0] = -2.0
    records[1, 1] = -2.0
    records[2, 0] = 1.0
    records[2, 1] = 1.0
    records[3, 0] = 2.0
    records[3, 1] = 2.0

    var targets = List[Scalar[DType.int32]]()
    targets.append(0)
    targets.append(0)
    targets.append(1)
    targets.append(1)

    var ds = Dataset[DType.float64, DType.int32](records^, targets^)
    var clf = LinearSVC[DType.float64]()

    fit_dataset[LinearSVC[DType.float64], DType.float64, DType.int32](clf, ds)
    assert_true(clf.is_fitted)

    var preds = predict_dataset[
        LinearSVC[DType.float64], DType.float64, DType.int32
    ](clf, ds)
    assert_equal(len(preds), 4)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_linear_svc_nan_inf_rejection() raises:
    var X_nan = Matrix[DType.float64](3, 2, 0)
    X_nan[0, 0] = nan[DType.float64]()
    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)

    var clf = LinearSVC[DType.float64]()
    with assert_raises():
        clf.fit(X_nan, y)

    var X_inf = Matrix[DType.float64](3, 2, 0)
    X_inf[0, 0] = inf[DType.float64]()
    with assert_raises():
        clf.fit(X_inf, y)


def test_linear_svc_parameter_validation() raises:
    # Invalid C
    with assert_raises():
        _ = LinearSVC[DType.float64](C=0.0)
    with assert_raises():
        _ = LinearSVC[DType.float64](C=-1.0)

    # Invalid loss
    with assert_raises():
        _ = LinearSVC[DType.float64](loss="logistic")

    # Invalid penalty
    with assert_raises():
        _ = LinearSVC[DType.float64](penalty="l1")

    # Invalid tol
    with assert_raises():
        _ = LinearSVC[DType.float64](tol=0.0)

    # Invalid intercept_scaling
    with assert_raises():
        _ = LinearSVC[DType.float64](intercept_scaling=-0.5)

    # Invalid max_iter
    with assert_raises():
        _ = LinearSVC[DType.float64](max_iter=0)

    # NotFittedError on decision_function / predict
    var unfitted = LinearSVC[DType.float64]()
    var dummy = Matrix[DType.float64](2, 2, 0)
    with assert_raises():
        _ = unfitted.decision_function(dummy)
    with assert_raises():
        _ = unfitted.predict(dummy)

    # Dimension mismatch on predict
    var X_train = Matrix[DType.float64](4, 2, 0)
    var y_train = List[Scalar[DType.int32]]()
    y_train.append(0)
    y_train.append(0)
    y_train.append(1)
    y_train.append(1)
    var fitted = LinearSVC[DType.float64]()
    fitted.fit(X_train, y_train)

    var X_wrong = Matrix[DType.float64](4, 3, 0)
    with assert_raises():
        _ = fitted.predict(X_wrong)

    # Only 1 class in y
    var y_single = List[Scalar[DType.int32]]()
    y_single.append(0)
    y_single.append(0)
    y_single.append(0)
    y_single.append(0)
    var clf_bad = LinearSVC[DType.float64]()
    with assert_raises():
        clf_bad.fit(X_train, y_single)


def test_linear_svc_repeated_fit_resets_state() raises:
    var X1 = Matrix[DType.float64](4, 2, 0)
    X1[0, 0] = -1.0
    X1[0, 1] = -1.0
    X1[1, 0] = -2.0
    X1[1, 1] = -2.0
    X1[2, 0] = 1.0
    X1[2, 1] = 1.0
    X1[3, 0] = 2.0
    X1[3, 1] = 2.0

    var y1 = List[Scalar[DType.int32]]()
    y1.append(0)
    y1.append(0)
    y1.append(1)
    y1.append(1)

    var clf = LinearSVC[DType.float64]()
    clf.fit(X1, y1)
    assert_equal(len(clf.classes_), 2)

    # Re-fit on 3-class dataset in 3 dimensions
    var X2 = Matrix[DType.float64](6, 3, 0)
    var y2 = List[Scalar[DType.int32]]()
    for i in range(6):
        y2.append(Scalar[DType.int32](i // 2))

    clf.fit(X2, y2)
    assert_equal(len(clf.classes_), 3)
    assert_equal(clf.n_features_in_, 3)
    assert_equal(clf.coef_.rows, 3)
    assert_equal(clf.coef_.cols, 3)
    assert_equal(len(clf.intercept_), 3)


def test_linear_svc_single_feature() raises:
    # 1D feature matrix (shape (6, 1))
    var X = Matrix[DType.float64](6, 1, 0)
    X[0, 0] = -5.0
    X[1, 0] = -4.0
    X[2, 0] = -3.0
    X[3, 0] = 3.0
    X[4, 0] = 4.0
    X[5, 0] = 5.0

    var y = List[Scalar[DType.int32]]()
    for _ in range(3):
        y.append(0)
    for _ in range(3):
        y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    assert_equal(clf.n_features_in_, 1)
    assert_equal(clf.coef_.cols, 1)

    var preds = clf.predict(X)
    for i in range(3):
        assert_equal(preds[i], 0)
    for i in range(3, 6):
        assert_equal(preds[i], 1)


def test_linear_svc_large_intercept_offset() raises:
    # Points centered around (100, 100) and (105, 105)
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 100.0
    X[0, 1] = 100.0
    X[1, 0] = 100.1
    X[1, 1] = 99.9
    X[2, 0] = 99.9
    X[2, 1] = 100.1

    X[3, 0] = 105.0
    X[3, 1] = 105.0
    X[4, 0] = 105.1
    X[4, 1] = 104.9
    X[5, 0] = 104.9
    X[5, 1] = 105.1

    var y = List[Scalar[DType.int32]]()
    for _ in range(3):
        y.append(0)
    for _ in range(3):
        y.append(1)

    var clf = LinearSVC[DType.float64](
        C=100.0, intercept_scaling=100.0, max_iter=2000
    )
    clf.fit(X, y)

    var preds = clf.predict(X)
    for i in range(3):
        assert_equal(preds[i], 0)
    for i in range(3, 6):
        assert_equal(preds[i], 1)


def test_linear_svc_unbalanced_classes() raises:
    # 8 samples in class 0, 2 samples in class 1
    var X = Matrix[DType.float64](10, 2, 0)
    var y = List[Scalar[DType.int32]](capacity=10)

    for i in range(8):
        X[i, 0] = -2.0 + Float64(i) * 0.1
        X[i, 1] = -2.0
        y.append(0)

    for i in range(2):
        X[8 + i, 0] = 5.0 + Float64(i) * 0.5
        X[8 + i, 1] = 5.0
        y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    var preds = clf.predict(X)
    for i in range(8):
        assert_equal(preds[i], 0)
    for i in range(8, 10):
        assert_equal(preds[i], 1)


def test_linear_svc_arbitrary_class_labels() raises:
    # Non-consecutive and negative class labels: -5 and 42
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -1.0
    X[1, 1] = -1.0
    X[2, 0] = 2.0
    X[2, 1] = 2.0
    X[3, 0] = 3.0
    X[3, 1] = 3.0

    var y = List[Scalar[DType.int32]]()
    y.append(-5)
    y.append(-5)
    y.append(42)
    y.append(42)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 2)
    assert_equal(clf.classes_[0], -5)
    assert_equal(clf.classes_[1], 42)

    var preds = clf.predict(X)
    assert_equal(preds[0], -5)
    assert_equal(preds[1], -5)
    assert_equal(preds[2], 42)
    assert_equal(preds[3], 42)


def test_linear_svc_multiclass_ovr_4class() raises:
    # 4 distinct quadrants: (5, 5), (-5, 5), (-5, -5), (5, -5)
    var X = Matrix[DType.float64](8, 2, 0)
    # Class 0: (+5, +5)
    X[0, 0] = 5.0
    X[0, 1] = 5.0
    X[1, 0] = 5.1
    X[1, 1] = 4.9
    # Class 1: (-5, +5)
    X[2, 0] = -5.0
    X[2, 1] = 5.0
    X[3, 0] = -5.1
    X[3, 1] = 4.9
    # Class 2: (-5, -5)
    X[4, 0] = -5.0
    X[4, 1] = -5.0
    X[5, 0] = -4.9
    X[5, 1] = -5.1
    # Class 3: (+5, -5)
    X[6, 0] = 5.0
    X[6, 1] = -5.0
    X[7, 0] = 5.1
    X[7, 1] = -4.9

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)
    y.append(2)
    y.append(2)
    y.append(3)
    y.append(3)

    var clf = LinearSVC[DType.float64](C=1.0, max_iter=2000)
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 4)
    assert_equal(clf.coef_.rows, 4)
    assert_equal(len(clf.intercept_), 4)

    var preds = clf.predict(X)
    for i in range(4):
        assert_equal(preds[2 * i], i)
        assert_equal(preds[2 * i + 1], i)


def test_linear_svc_zero_variance_feature() raises:
    # Feature 0 is informative, Feature 1 is a constant 1.0 for all samples
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = 1.0
    X[1, 0] = -1.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[3, 0] = 2.0
    X[3, 1] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    var clf = LinearSVC[DType.float64](C=1.0)
    clf.fit(X, y)

    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_linear_svc_max_iter_early_stop() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -1.0
    X[0, 1] = -1.0
    X[1, 0] = -2.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[3, 0] = 2.0
    X[3, 1] = 2.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)

    # Even with max_iter=2, training completes gracefully without crashing
    var clf = LinearSVC[DType.float64](max_iter=2)
    clf.fit(X, y)
    assert_true(clf.is_fitted)
    assert_true(clf.n_iter_ <= 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
