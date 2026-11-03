from std.math import abs, sin, nan, inf
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
from strata.svm.svr import SVR
from strata.preprocessing.scaler import StandardScaler
from strata.base.pipeline import PipelineRegressor
from strata.io.serializer import dumps, loads


def test_svr_1d_linear() raises:
    var X = Matrix[DType.float64](5, 1, 0.0)
    for i in range(5):
        X[i, 0] = Float64(i + 1)

    var y: List[Scalar[DType.float64]] = [3.0, 5.0, 7.0, 9.0, 11.0]

    var reg = SVR[DType.float64](
        C=10.0, epsilon=0.01, kernel="linear", tol=1e-4
    )
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.n_features_in_, 1)
    assert_true(reg.support_vectors_.rows > 0)
    assert_equal(reg.support_vectors_.cols, 1)

    var preds = reg.predict(X)
    assert_equal(len(preds), 5)
    for i in range(5):
        assert_almost_equal(preds[i], y[i], atol=0.1)

    # Without intercept: y = 3 * x
    var y_no_bias: List[Scalar[DType.float64]] = [3.0, 6.0, 9.0, 12.0, 15.0]
    var reg_no_bias = SVR[DType.float64](
        C=10.0, epsilon=0.01, kernel="linear", fit_intercept=False, tol=1e-4
    )
    reg_no_bias.fit(X, y_no_bias)
    assert_equal(reg_no_bias.intercept_, 0.0)
    var preds_nb = reg_no_bias.predict(X)
    for i in range(5):
        assert_almost_equal(preds_nb[i], y_no_bias[i], atol=0.15)


def test_svr_nonlinear_quadratic() raises:
    # y = x^2 on [-2, -1, 0, 1, 2]
    var X = Matrix[DType.float64](5, 1, 0.0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 0.0
    X[3, 0] = 1.0
    X[4, 0] = 2.0

    var y: List[Scalar[DType.float64]] = [4.0, 1.0, 0.0, 1.0, 4.0]

    var reg_rbf = SVR[DType.float64](
        C=20.0, epsilon=0.01, kernel="rbf", gamma_val=0.5, tol=1e-4
    )
    reg_rbf.fit(X, y)

    var preds_rbf = reg_rbf.predict(X)
    for i in range(5):
        assert_almost_equal(preds_rbf[i], y[i], atol=0.15)

    var reg_poly = SVR[DType.float64](
        C=10.0,
        epsilon=0.01,
        kernel="poly",
        degree=2,
        gamma_val=1.0,
        coef0=0.0,
        tol=1e-4,
    )
    reg_poly.fit(X, y)

    var preds_poly = reg_poly.predict(X)
    for i in range(5):
        assert_almost_equal(preds_poly[i], y[i], atol=0.15)


def test_svr_nonlinear_sine_curve() raises:
    var n = 9
    var X = Matrix[DType.float64](n, 1, 0.0)
    var y = List[Scalar[DType.float64]](capacity=n)

    for i in range(n):
        var x_val = Float64(i) * 0.5
        X[i, 0] = x_val
        y.append(sin(x_val))

    var reg = SVR[DType.float64](
        C=10.0, epsilon=0.01, kernel="rbf", gamma_val=0.5, tol=1e-4
    )
    reg.fit(X, y)

    var preds = reg.predict(X)
    for i in range(n):
        assert_almost_equal(preds[i], y[i], atol=0.1)

    # Midpoint interpolation at x = 0.75 and x = 1.75
    var X_test = Matrix[DType.float64](2, 1, 0.0)
    X_test[0, 0] = 0.75
    X_test[1, 0] = 1.75
    var test_preds = reg.predict(X_test)
    assert_almost_equal(test_preds[0], sin(0.75), atol=0.15)
    assert_almost_equal(test_preds[1], sin(1.75), atol=0.15)


def test_svr_kernels_and_gamma_modes() raises:
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 0.5
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    X[1, 1] = 2.0
    X[2, 0] = 1.5
    X[2, 1] = 2.5
    X[3, 0] = 2.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [1.5, 3.0, 4.0, 5.0]

    var reg_auto = SVR[DType.float64](C=5.0, kernel="rbf", gamma="auto")
    reg_auto.fit(X, y)
    assert_almost_equal(reg_auto.gamma_, 0.5, atol=1e-6)
    var preds_auto = reg_auto.predict(X)
    assert_equal(len(preds_auto), 4)

    var reg_sig = SVR[DType.float64](
        C=5.0,
        epsilon=0.05,
        kernel="sigmoid",
        gamma_val=0.05,
        coef0=0.0,
    )
    reg_sig.fit(X, y)
    assert_true(reg_sig.is_fitted)
    var preds_sig = reg_sig.predict(X)
    assert_equal(len(preds_sig), 4)
    assert_true(preds_sig[3] > preds_sig[0])


def test_svr_epsilon_tube_and_empty_sv() raises:
    var X = Matrix[DType.float64](4, 1, 0.0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    # Targets within [-0.2, 0.2]; epsilon = 1.0 covers all points at beta = 0
    var y: List[Scalar[DType.float64]] = [0.1, -0.1, 0.15, -0.05]
    var reg_wide = SVR[DType.float64](C=1.0, epsilon=1.0, kernel="rbf")
    reg_wide.fit(X, y)

    assert_equal(reg_wide.support_vectors_.rows, 0)
    var preds_wide = reg_wide.predict(X)
    for i in range(4):
        assert_almost_equal(preds_wide[i], 0.0, atol=1e-6)


def test_svr_multivariate_and_high_dimensional() raises:
    # 3D features: y = x0 + 2*x1 - x2
    var X_3d = Matrix[DType.float64](6, 3, 0.0)
    X_3d[0, 0] = 1.0
    X_3d[0, 1] = 0.0
    X_3d[0, 2] = 1.0
    X_3d[1, 0] = 0.0
    X_3d[1, 1] = 1.0
    X_3d[1, 2] = 0.0
    X_3d[2, 0] = 2.0
    X_3d[2, 1] = 1.0
    X_3d[2, 2] = 1.0
    X_3d[3, 0] = 1.0
    X_3d[3, 1] = 2.0
    X_3d[3, 2] = 0.0
    X_3d[4, 0] = 0.0
    X_3d[4, 1] = 2.0
    X_3d[4, 2] = 2.0
    X_3d[5, 0] = 2.0
    X_3d[5, 1] = 0.0
    X_3d[5, 2] = 0.0

    var y_3d = List[Scalar[DType.float64]](capacity=6)
    for i in range(6):
        y_3d.append(X_3d[i, 0] + 2.0 * X_3d[i, 1] - X_3d[i, 2])

    var reg_3d = SVR[DType.float64](
        C=10.0, epsilon=0.01, kernel="linear", tol=1e-4
    )
    reg_3d.fit(X_3d, y_3d)
    var preds_3d = reg_3d.predict(X_3d)
    for i in range(6):
        assert_almost_equal(preds_3d[i], y_3d[i], atol=0.15)

    # High-dimensional features (D = 12 > N = 5)
    var X_hd = Matrix[DType.float64](5, 12, 0.0)
    var y_hd = List[Scalar[DType.float64]](capacity=5)
    for i in range(5):
        for j in range(12):
            X_hd[i, j] = Float64(i * 2 + j) * 0.1
        y_hd.append(Float64(i + 1) * 1.5)

    var reg_hd = SVR[DType.float64](C=10.0, epsilon=0.01, kernel="rbf")
    reg_hd.fit(X_hd, y_hd)

    assert_equal(reg_hd.n_features_in_, 12)
    assert_equal(reg_hd.support_vectors_.cols, 12)
    var preds_hd = reg_hd.predict(X_hd)
    assert_equal(len(preds_hd), 5)
    for i in range(5):
        assert_almost_equal(preds_hd[i], y_hd[i], atol=0.2)


def test_svr_float32_dataset_and_pipeline() raises:
    var X_f32 = Matrix[DType.float32](4, 1, 0.0)
    X_f32[0, 0] = 1.0
    X_f32[1, 0] = 2.0
    X_f32[2, 0] = 3.0
    X_f32[3, 0] = 4.0
    var y_f32: List[Scalar[DType.float32]] = [2.0, 4.0, 6.0, 8.0]

    var reg_f32 = SVR[DType.float32](
        C=10.0, epsilon=0.01, kernel="linear", tol=1e-3
    )
    var ds = Dataset[DType.float32, DType.float32](X_f32.copy(), y_f32.copy())
    reg_f32.fit(ds)
    var preds_f32 = reg_f32.predict(ds)
    assert_equal(len(preds_f32), 4)
    for i in range(4):
        assert_almost_equal(Float64(preds_f32[i]), Float64(y_f32[i]), atol=0.15)

    # PipelineRegressor integration
    var X_pipe = Matrix[DType.float64](4, 2, 0.0)
    X_pipe[0, 0] = 10.0
    X_pipe[0, 1] = 100.0
    X_pipe[1, 0] = 20.0
    X_pipe[1, 1] = 200.0
    X_pipe[2, 0] = 30.0
    X_pipe[2, 1] = 300.0
    X_pipe[3, 0] = 40.0
    X_pipe[3, 1] = 400.0
    var y_pipe: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    var pipe = PipelineRegressor(
        StandardScaler(), SVR[DType.float64](C=10.0, epsilon=0.01, kernel="rbf")
    )
    pipe.fit(X_pipe, y_pipe)
    var pipe_preds = pipe.predict(X_pipe)
    for i in range(4):
        assert_almost_equal(pipe_preds[i], y_pipe[i], atol=0.15)


def test_svr_copy_and_serialization() raises:
    var X = Matrix[DType.float64](5, 2, 0.0)
    for i in range(5):
        X[i, 0] = Float64(i)
        X[i, 1] = Float64(i) * 0.5
    var y: List[Scalar[DType.float64]] = [1.0, 1.5, 3.0, 5.5, 9.0]

    var reg = SVR[DType.float64](
        C=10.0, epsilon=0.02, kernel="rbf", gamma_val=0.5
    )
    reg.fit(X, y)

    var reg_copy = reg.copy()
    var bytes = dumps(reg)
    var loaded = loads[SVR[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_equal(loaded.kernel, reg.kernel)
    assert_almost_equal(loaded.C, reg.C)
    assert_almost_equal(loaded.epsilon, reg.epsilon)
    assert_almost_equal(loaded.gamma_, reg.gamma_)
    assert_almost_equal(loaded.intercept_, reg.intercept_)
    assert_equal(loaded.support_vectors_.rows, reg.support_vectors_.rows)
    assert_equal(loaded.support_vectors_.cols, reg.support_vectors_.cols)
    assert_equal(len(loaded.dual_coef_), len(reg.dual_coef_))

    var orig_preds = reg.predict(X)
    var copy_preds = reg_copy.predict(X)
    var loaded_preds = loaded.predict(X)
    for i in range(5):
        assert_almost_equal(orig_preds[i], copy_preds[i], atol=1e-12)
        assert_almost_equal(orig_preds[i], loaded_preds[i], atol=1e-12)


def test_svr_errors() raises:
    var reg = SVR[DType.float64]()
    var X = Matrix[DType.float64](3, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]

    # Unfitted predict
    with assert_raises():
        _ = reg.predict(X)

    # Invalid hyperparameters
    var bad_c = SVR[DType.float64](C=0.0)
    with assert_raises():
        bad_c.fit(X, y)

    var bad_eps = SVR[DType.float64](epsilon=-0.1)
    with assert_raises():
        bad_eps.fit(X, y)

    var bad_tol = SVR[DType.float64](tol=0.0)
    with assert_raises():
        bad_tol.fit(X, y)

    var bad_deg = SVR[DType.float64](kernel="poly", degree=0)
    with assert_raises():
        bad_deg.fit(X, y)

    var bad_iter = SVR[DType.float64](max_iter=0)
    with assert_raises():
        bad_iter.fit(X, y)

    var bad_kernel = SVR[DType.float64](kernel="cosine")
    with assert_raises():
        bad_kernel.fit(X, y)

    var bad_gamma = SVR[DType.float64](gamma="invalid_gamma")
    with assert_raises():
        bad_gamma.fit(X, y)

    # Sample length mismatch
    var y_short: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        reg.fit(X, y_short)

    # NaN / Inf inputs
    var X_nan = X.copy()
    X_nan[0, 0] = nan[DType.float64]()
    with assert_raises():
        reg.fit(X_nan, y)

    var X_inf = X.copy()
    X_inf[0, 0] = inf[DType.float64]()
    with assert_raises():
        reg.fit(X_inf, y)

    # Feature count mismatch on predict
    reg.fit(X, y)
    var X_bad_cols = Matrix[DType.float64](2, 4, 1.0)
    with assert_raises():
        _ = reg.predict(X_bad_cols)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
