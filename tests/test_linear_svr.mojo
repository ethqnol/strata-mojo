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
from strata.svm.linear_svr import LinearSVR
from strata.preprocessing.scaler import StandardScaler
from strata.base.pipeline import PipelineRegressor
from strata.io.serializer import dumps, loads
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_linear_svr_basic_1d() raises:
    # y = 2.0 * x + 1.0
    var X = Matrix[DType.float64](5, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    X[4, 0] = 4.0

    var y = List[Scalar[DType.float64]](capacity=5)
    y.append(1.0)
    y.append(3.0)
    y.append(5.0)
    y.append(7.0)
    y.append(9.0)

    var reg = LinearSVR[DType.float64](
        epsilon=0.0, C=10.0, loss="epsilon_insensitive", tol=1e-5
    )
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.n_features_in_, 1)
    assert_equal(len(reg.coef_), 1)
    assert_almost_equal(reg.coef_[0], 2.0, rtol=1e-2)
    assert_almost_equal(reg.intercept_, 1.0, rtol=1e-2)

    var preds = reg.predict(X)
    assert_equal(len(preds), 5)
    for i in range(5):
        assert_almost_equal(preds[i], y[i], rtol=1e-2)


def test_linear_svr_multivariate() raises:
    # y = 1.5 * x0 - 2.0 * x1 + 0.5 * x2 + 3.0
    var X = Matrix[DType.float64](6, 3, 0)
    X[0, 0] = 0.0
    X[0, 1] = 1.0
    X[0, 2] = 2.0
    X[1, 0] = 1.0
    X[1, 1] = 0.0
    X[1, 2] = 1.0
    X[2, 0] = 2.0
    X[2, 1] = 2.0
    X[2, 2] = 0.0
    X[3, 0] = 3.0
    X[3, 1] = 1.0
    X[3, 2] = 3.0
    X[4, 0] = 1.0
    X[4, 1] = 3.0
    X[4, 2] = 2.0
    X[5, 0] = 4.0
    X[5, 1] = 0.0
    X[5, 2] = 1.0

    var y = List[Scalar[DType.float64]](capacity=6)
    for i in range(6):
        var val = 1.5 * X[i, 0] - 2.0 * X[i, 1] + 0.5 * X[i, 2] + 3.0
        y.append(val)

    var reg = LinearSVR[DType.float64](
        epsilon=0.0, C=10.0, loss="epsilon_insensitive", tol=1e-5
    )
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.n_features_in_, 3)
    assert_equal(len(reg.coef_), 3)

    var preds = reg.predict(X)
    for i in range(6):
        assert_almost_equal(preds[i], y[i], rtol=5e-2)


def test_linear_svr_epsilon_insensitive_l1() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 3.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 5.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(3.0)
    y.append(5.0)
    y.append(7.0)
    y.append(9.0)

    var reg = LinearSVR[DType.float64](
        epsilon=0.1, C=1.0, loss="epsilon_insensitive"
    )
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    var preds = reg.predict(X)
    for i in range(4):
        assert_almost_equal(preds[i], y[i], rtol=1e-1)


def test_linear_svr_squared_epsilon_insensitive_l2() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 3.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 5.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(3.0)
    y.append(5.0)
    y.append(7.0)
    y.append(9.0)

    var reg = LinearSVR[DType.float64](
        epsilon=0.1, C=1.0, loss="squared_epsilon_insensitive"
    )
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    var preds = reg.predict(X)
    for i in range(4):
        assert_almost_equal(preds[i], y[i], rtol=1e-1)


def test_linear_svr_epsilon_tube_insensitivity() raises:
    # Within epsilon tube, small perturbation should not change solution significantly
    var X = Matrix[DType.float64](5, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    X[4, 0] = 4.0

    var y1 = List[Scalar[DType.float64]](capacity=5)
    y1.append(1.0)
    y1.append(3.0)
    y1.append(5.0)
    y1.append(7.0)
    y1.append(9.0)

    var reg1 = LinearSVR[DType.float64](
        epsilon=1.0, C=1.0, loss="epsilon_insensitive"
    )
    reg1.fit(X, y1)

    # Perturb middle point by 0.2 (well within epsilon=1.0)
    var y2 = List[Scalar[DType.float64]](capacity=5)
    y2.append(1.0)
    y2.append(3.0)
    y2.append(5.2)
    y2.append(7.0)
    y2.append(9.0)

    var reg2 = LinearSVR[DType.float64](
        epsilon=1.0, C=1.0, loss="epsilon_insensitive"
    )
    reg2.fit(X, y2)

    var p1 = reg1.predict(X)
    var p2 = reg2.predict(X)
    for i in range(5):
        assert_almost_equal(p1[i], p2[i], rtol=5e-2)


def test_linear_svr_no_intercept() raises:
    # y = 3.0 * x through origin
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(3.0)
    y.append(6.0)
    y.append(9.0)
    y.append(12.0)

    var reg = LinearSVR[DType.float64](
        epsilon=0.0, C=10.0, fit_intercept=False, tol=1e-5
    )
    reg.fit(X, y)

    assert_equal(reg.intercept_, 0.0)
    assert_almost_equal(reg.coef_[0], 3.0, rtol=1e-2)


def test_linear_svr_intercept_scaling() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(5.0)
    y.append(7.0)
    y.append(9.0)
    y.append(11.0)

    var reg = LinearSVR[DType.float64](
        epsilon=0.0, C=10.0, fit_intercept=True, intercept_scaling=5.0, tol=1e-5
    )
    reg.fit(X, y)

    var preds = reg.predict(X)
    for i in range(4):
        assert_almost_equal(preds[i], y[i], rtol=5e-2)


def test_linear_svr_c_regularization() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(10.0)
    y.append(20.0)
    y.append(30.0)
    y.append(40.0)

    # Large C: fits slope ~ 10
    var reg_large_c = LinearSVR[DType.float64](
        epsilon=0.0, C=100.0, fit_intercept=False
    )
    reg_large_c.fit(X, y)

    # Very small C: heavily shrinks weights toward 0
    var reg_small_c = LinearSVR[DType.float64](
        epsilon=0.0, C=0.001, fit_intercept=False
    )
    reg_small_c.fit(X, y)

    assert_true(abs(reg_large_c.coef_[0]) > abs(reg_small_c.coef_[0]))


def test_linear_svr_dataset_interop() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.5
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 1.5
    X[3, 0] = 4.0
    X[3, 1] = 2.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(2.0)
    y.append(4.0)
    y.append(6.0)
    y.append(8.0)

    var ds = Dataset[DType.float64, DType.float64](X.copy(), y.copy())
    var reg = LinearSVR[DType.float64](epsilon=0.0, C=1.0)
    reg.fit(ds)

    assert_true(reg.is_fitted)
    var preds = reg.predict(ds)
    assert_equal(len(preds), 4)


def test_linear_svr_pipeline_interop() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 10.0
    X[0, 1] = 200.0
    X[1, 0] = 20.0
    X[1, 1] = 300.0
    X[2, 0] = 30.0
    X[2, 1] = 400.0
    X[3, 0] = 40.0
    X[3, 1] = 500.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(1.0)
    y.append(2.0)
    y.append(3.0)
    y.append(4.0)

    var scaler = StandardScaler()
    var svr = LinearSVR[DType.float64](C=1.0)
    var pipe = PipelineRegressor(scaler^, svr^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)
    assert_almost_equal(preds[0], 1.0, rtol=1e-1)


def test_linear_svr_float32() raises:
    var X = Matrix[DType.float32](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var y = List[Scalar[DType.float32]](capacity=4)
    y.append(2.0)
    y.append(4.0)
    y.append(6.0)
    y.append(8.0)

    var reg = LinearSVR[DType.float32](epsilon=0.0, C=5.0)
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    var preds = reg.predict(X)
    assert_equal(len(preds), 4)
    assert_almost_equal(Float64(preds[0]), 2.0, rtol=5e-2)


def test_linear_svr_copy_semantics() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(1.0)
    y.append(2.0)
    y.append(3.0)
    y.append(4.0)

    var reg1 = LinearSVR[DType.float64](C=1.0)
    reg1.fit(X, y)

    var reg2 = reg1.copy()
    assert_true(reg2.is_fitted)
    assert_equal(len(reg2.coef_), 1)
    assert_almost_equal(reg2.coef_[0], reg1.coef_[0])

    # Modify reg2 by refitting on different data
    var y2 = List[Scalar[DType.float64]](capacity=4)
    y2.append(10.0)
    y2.append(20.0)
    y2.append(30.0)
    y2.append(40.0)
    reg2.fit(X, y2)

    # reg1 should remain untouched
    assert_almost_equal(reg1.coef_[0], 1.0, rtol=1e-1)
    assert_true(reg2.coef_[0] > 5.0)
    assert_true(abs(reg1.coef_[0] - reg2.coef_[0]) > 2.0)


def test_linear_svr_serialization_roundtrip() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.5
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 1.5
    X[3, 0] = 4.0
    X[3, 1] = 2.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(3.0)
    y.append(5.0)
    y.append(7.0)
    y.append(9.0)

    var reg = LinearSVR[DType.float64](epsilon=0.1, C=2.0)
    reg.fit(X, y)

    var bytes = dumps(reg)
    var restored = loads[LinearSVR[DType.float64]](bytes)

    assert_true(restored.is_fitted)
    assert_equal(restored.n_features_in_, reg.n_features_in_)
    assert_almost_equal(restored.intercept_, reg.intercept_)
    assert_equal(len(restored.coef_), len(reg.coef_))
    for j in range(len(reg.coef_)):
        assert_almost_equal(restored.coef_[j], reg.coef_[j])

    var orig_preds = reg.predict(X)
    var restored_preds = restored.predict(X)
    for i in range(4):
        assert_almost_equal(orig_preds[i], restored_preds[i])


def test_linear_svr_nan_inf_rejection() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = nan[DType.float64]()
    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(1.0)
    y.append(2.0)
    y.append(3.0)
    y.append(4.0)

    var reg = LinearSVR[DType.float64]()
    with assert_raises():
        reg.fit(X, y)

    var X_inf = Matrix[DType.float64](4, 2, 0)
    X_inf[0, 0] = inf[DType.float64]()
    with assert_raises():
        reg.fit(X_inf, y)


def test_linear_svr_parameter_validation() raises:
    with assert_raises():
        _ = LinearSVR[DType.float64](C=-1.0)

    with assert_raises():
        _ = LinearSVR[DType.float64](C=0.0)

    with assert_raises():
        _ = LinearSVR[DType.float64](epsilon=-0.5)

    with assert_raises():
        _ = LinearSVR[DType.float64](tol=0.0)

    with assert_raises():
        _ = LinearSVR[DType.float64](max_iter=0)

    with assert_raises():
        _ = LinearSVR[DType.float64](intercept_scaling=-1.0)

    with assert_raises():
        _ = LinearSVR[DType.float64](loss="unsupported_loss")

    with assert_raises():
        _ = LinearSVR[DType.float64](dual=False)


def test_linear_svr_unfitted_errors() raises:
    var reg = LinearSVR[DType.float64]()
    var X = Matrix[DType.float64](2, 2, 0)

    with assert_raises():
        _ = reg.predict(X)


def test_linear_svr_feature_dimension_mismatch() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(1.0)
    y.append(2.0)
    y.append(3.0)
    y.append(4.0)

    var reg = LinearSVR[DType.float64]()
    reg.fit(X, y)

    var X_bad = Matrix[DType.float64](4, 3, 0)
    with assert_raises():
        _ = reg.predict(X_bad)


def test_linear_svr_length_mismatch() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=3)
    y.append(1.0)
    y.append(2.0)
    y.append(3.0)

    var reg = LinearSVR[DType.float64]()
    with assert_raises():
        reg.fit(X, y)


def test_linear_svr_single_feature() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y = List[Scalar[DType.float64]](capacity=3)
    y.append(2.0)
    y.append(4.0)
    y.append(6.0)

    var reg = LinearSVR[DType.float64](C=10.0, tol=1e-5)
    reg.fit(X, y)

    assert_equal(reg.n_features_in_, 1)
    assert_equal(len(reg.coef_), 1)
    var preds = reg.predict(X)
    assert_almost_equal(preds[0], 2.0, rtol=1e-2)


def test_linear_svr_high_dimensional() raises:
    # 4 samples with 8 features (underdetermined D > N)
    var X = Matrix[DType.float64](4, 8, 0)
    for r in range(4):
        for c in range(8):
            X[r, c] = Float64(r + c)

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(1.0)
    y.append(2.0)
    y.append(3.0)
    y.append(4.0)

    var reg = LinearSVR[DType.float64](C=1.0)
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.n_features_in_, 8)
    assert_equal(len(reg.coef_), 8)
    var preds = reg.predict(X)
    assert_equal(len(preds), 4)


def test_linear_svr_refitting_state_reset() raises:
    var X1 = Matrix[DType.float64](4, 2, 1.0)
    var y1 = List[Scalar[DType.float64]](capacity=4)
    y1.append(1.0)
    y1.append(1.0)
    y1.append(1.0)
    y1.append(1.0)

    var reg = LinearSVR[DType.float64]()
    reg.fit(X1, y1)
    assert_equal(reg.n_features_in_, 2)
    assert_equal(len(reg.coef_), 2)

    var X2 = Matrix[DType.float64](4, 4, 1.0)
    var y2 = List[Scalar[DType.float64]](capacity=4)
    y2.append(2.0)
    y2.append(2.0)
    y2.append(2.0)
    y2.append(2.0)

    reg.fit(X2, y2)
    assert_equal(reg.n_features_in_, 4)
    assert_equal(len(reg.coef_), 4)


def test_linear_svr_collinear_zero_variance() raises:
    # Column 1 has 0 variance (all 0.0)
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 0.0
    X[2, 0] = 3.0
    X[2, 1] = 0.0
    X[3, 0] = 4.0
    X[3, 1] = 0.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(2.0)
    y.append(4.0)
    y.append(6.0)
    y.append(8.0)

    var reg = LinearSVR[DType.float64](C=10.0, tol=1e-5)
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_almost_equal(reg.coef_[1], 0.0, atol=1e-5)
    var preds = reg.predict(X)
    assert_almost_equal(preds[0], 2.0, rtol=1e-2)


def test_linear_svr_constant_target() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y = List[Scalar[DType.float64]](capacity=4)
    y.append(5.0)
    y.append(5.0)
    y.append(5.0)
    y.append(5.0)

    var reg = LinearSVR[DType.float64](C=10.0, tol=1e-5)
    reg.fit(X, y)

    var preds = reg.predict(X)
    for i in range(4):
        assert_almost_equal(preds[i], 5.0, rtol=5e-2)


def test_linear_svr_noisy_outliers() raises:
    # Data with a single outlier
    var X = Matrix[DType.float64](6, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    X[4, 0] = 5.0
    X[5, 0] = 6.0

    var y = List[Scalar[DType.float64]](capacity=6)
    y.append(2.0)
    y.append(4.0)
    y.append(6.0)
    y.append(8.0)
    y.append(10.0)
    y.append(100.0)  # Extreme outlier at x=6

    var reg = LinearSVR[DType.float64](
        epsilon=0.1, C=1.0, loss="epsilon_insensitive"
    )
    reg.fit(X, y)

    # In linear SVR with L1 loss, the slope is resistant to the outlier
    assert_true(reg.coef_[0] < 20.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
