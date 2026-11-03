from std.math import abs, exp
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
from strata.svm.svc import SVC
from strata.io.serializer import dumps, loads
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_svc_binary_separable() raises:
    # 4 points for class 0, 4 points for class 1
    var X = Matrix[DType.float64](8, 2, 0.0)
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

    var clf = SVC[DType.float64](C=1.0, kernel="rbf")
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(len(clf.classes_), 2)
    assert_equal(clf.classes_[0], 0)
    assert_equal(clf.classes_[1], 1)
    assert_true(clf.support_vectors_.rows > 0)
    assert_equal(clf.support_vectors_.cols, 2)

    var preds = clf.predict(X)
    assert_equal(len(preds), 8)
    for i in range(4):
        assert_equal(preds[i], 0)
    for i in range(4, 8):
        assert_equal(preds[i], 1)

    # Probabilities
    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 8)
    assert_equal(probs.cols, 2)
    for i in range(8):
        var row_sum = probs[i, 0] + probs[i, 1]
        assert_almost_equal(row_sum, 1.0, atol=1e-5)
        if i < 4:
            assert_true(probs[i, 0] > probs[i, 1])
        else:
            assert_true(probs[i, 1] > probs[i, 0])


def test_svc_non_linear_concentric_circles() raises:
    # Inner circle (radius ~0.5): class 0
    # Outer circle (radius ~2.0): class 1
    var X = Matrix[DType.float64](8, 2, 0.0)
    # Inner circle points
    X[0, 0] = 0.5
    X[0, 1] = 0.0
    X[1, 0] = -0.5
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 0.5
    X[3, 0] = 0.0
    X[3, 1] = -0.5

    # Outer circle points
    X[4, 0] = 2.0
    X[4, 1] = 0.0
    X[5, 0] = -2.0
    X[5, 1] = 0.0
    X[6, 0] = 0.0
    X[6, 1] = 2.0
    X[7, 0] = 0.0
    X[7, 1] = -2.0

    var y = List[Scalar[DType.int32]](capacity=8)
    for _ in range(4):
        y.append(0)
    for _ in range(4):
        y.append(1)

    # RBF non-linear kernel easily separates concentric circles
    var clf = SVC[DType.float64](C=10.0, kernel="rbf", gamma_val=1.0)
    clf.fit(X, y)

    var preds = clf.predict(X)
    for i in range(4):
        assert_equal(preds[i], 0)
    for i in range(4, 8):
        assert_equal(preds[i], 1)


def test_svc_xor_non_linear() raises:
    # XOR problem
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 1.0
    X[1, 1] = 1.0
    X[2, 0] = 0.0
    X[2, 1] = 1.0
    X[3, 0] = 1.0
    X[3, 1] = 0.0

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var clf = SVC[DType.float64](C=10.0, kernel="rbf", gamma_val=2.0)
    clf.fit(X, y)

    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_svc_multiclass_ovr() raises:
    # 3 clusters in 2D
    var X = Matrix[DType.float64](6, 2, 0.0)
    # Class 0: near (-3, 0)
    X[0, 0] = -3.0
    X[0, 1] = 0.0
    X[1, 0] = -3.2
    X[1, 1] = 0.1
    # Class 1: near (0, 3)
    X[2, 0] = 0.0
    X[2, 1] = 3.0
    X[3, 0] = 0.1
    X[3, 1] = 3.1
    # Class 2: near (3, 0)
    X[4, 0] = 3.0
    X[4, 1] = 0.0
    X[5, 0] = 3.1
    X[5, 1] = -0.1

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1, 2, 2]

    var clf = SVC[DType.float64](C=1.0, kernel="rbf")
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 3)
    var dec = clf.decision_function(X)
    assert_equal(dec.rows, 6)
    assert_equal(dec.cols, 3)

    var preds = clf.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))

    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 6)
    assert_equal(probs.cols, 3)
    for i in range(6):
        var s = probs[i, 0] + probs[i, 1] + probs[i, 2]
        assert_almost_equal(s, 1.0, atol=1e-5)


def test_svc_poly_and_linear() raises:
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 2.0
    X[1, 1] = 2.0
    X[2, 0] = -1.0
    X[2, 1] = -1.0
    X[3, 0] = -2.0
    X[3, 1] = -2.0

    var y: List[Scalar[DType.int32]] = [1, 1, 0, 0]

    # Linear kernel
    var clf_lin = SVC[DType.float64](C=1.0, kernel="linear")
    clf_lin.fit(X, y)
    var preds_lin = clf_lin.predict(X)
    assert_equal(preds_lin[0], 1)
    assert_equal(preds_lin[2], 0)

    # Poly kernel
    var clf_poly = SVC[DType.float64](C=1.0, kernel="poly", degree=3)
    clf_poly.fit(X, y)
    var preds_poly = clf_poly.predict(X)
    assert_equal(preds_poly[0], 1)
    assert_equal(preds_poly[2], 0)


def test_svc_serialization() raises:
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 2.0
    X[1, 1] = 2.0
    X[2, 0] = -1.0
    X[2, 1] = -1.0
    X[3, 0] = -2.0
    X[3, 1] = -2.0

    var y: List[Scalar[DType.int32]] = [1, 1, 0, 0]

    var clf = SVC[DType.float64](C=1.0, kernel="rbf")
    clf.fit(X, y)

    var bytes = dumps(clf)
    var loaded = loads[SVC[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_equal(loaded.kernel, clf.kernel)
    assert_equal(len(loaded.classes_), len(clf.classes_))
    assert_equal(loaded.support_vectors_.rows, clf.support_vectors_.rows)

    var orig_preds = clf.predict(X)
    var loaded_preds = loaded.predict(X)
    for i in range(4):
        assert_equal(orig_preds[i], loaded_preds[i])


def test_svc_errors() raises:
    var clf = SVC[DType.float64]()
    var X = Matrix[DType.float64](2, 2, 1.0)
    var y: List[Scalar[DType.int32]] = [0, 1]

    # Unfitted predict raises NotFittedError
    with assert_raises():
        _ = clf.predict(X)

    # Invalid C
    var bad_clf = SVC[DType.float64](C=-1.0)
    with assert_raises():
        bad_clf.fit(X, y)

    # Single class target raises InvalidParameterError
    var single_class_y: List[Scalar[DType.int32]] = [1, 1]
    with assert_raises():
        clf.fit(X, single_class_y)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
