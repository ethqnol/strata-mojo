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
from strata.svm._kernels import compute_gram_matrix, kernel_eval_rows
from strata.svm._smo import _smo_binary_svc, _kernel_svr_solver
from strata.exceptions.errors import (
    InvalidParameterError,
    DimensionMismatchError,
)


def test_smo_binary_separable() raises:
    # 4 samples: 2 positive at (2, 2), (3, 3); 2 negative at (-2, -2), (-3, -3)
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 2.0
    X[0, 1] = 2.0
    X[1, 0] = 3.0
    X[1, 1] = 3.0
    X[2, 0] = -2.0
    X[2, 1] = -2.0
    X[3, 0] = -3.0
    X[3, 1] = -3.0

    var y: List[Scalar[DType.float64]] = [1.0, 1.0, -1.0, -1.0]

    var K = compute_gram_matrix(X, "rbf", gamma=0.1)
    var res = _smo_binary_svc(K, y, C=1.0, tol=1e-4)

    assert_true(res.converged)
    assert_true(len(res.support_indices) > 0)

    # Check predictions: f(x_i) = sum_m dual_coef[m] * K(sv_m, x_i) + b
    for i in range(4):
        var pred: Scalar[DType.float64] = 0.0
        for m in range(len(res.support_indices)):
            var sv_idx = res.support_indices[m]
            var k_val = K[sv_idx, i]
            pred += res.dual_coef[m] * k_val
        pred += res.intercept

        if y[i] > 0:
            assert_true(pred > 0.0)
        else:
            assert_true(pred < 0.0)


def test_smo_xor_non_linear() raises:
    # Classic XOR problem:
    # (0, 0) -> -1, (1, 1) -> -1
    # (0, 1) -> +1, (1, 0) -> +1
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 1.0
    X[1, 1] = 1.0
    X[2, 0] = 0.0
    X[2, 1] = 1.0
    X[3, 0] = 1.0
    X[3, 1] = 0.0

    var y: List[Scalar[DType.float64]] = [-1.0, -1.0, 1.0, 1.0]

    # RBF with gamma=2.0 separates XOR perfectly
    var K = compute_gram_matrix(X, "rbf", gamma=2.0)
    var res = _smo_binary_svc(K, y, C=10.0, tol=1e-4)

    assert_true(res.converged)

    for i in range(4):
        var pred: Scalar[DType.float64] = 0.0
        for m in range(len(res.support_indices)):
            var sv_idx = res.support_indices[m]
            pred += res.dual_coef[m] * K[sv_idx, i]
        pred += res.intercept

        if y[i] > 0:
            assert_true(pred > 0.0)
        else:
            assert_true(pred < 0.0)


def test_kernel_svr_linear() raises:
    # y = 2 * x + 1 for x in {1, 2, 3, 4}
    var X = Matrix[DType.float64](4, 1, 0.0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var y: List[Scalar[DType.float64]] = [3.0, 5.0, 7.0, 9.0]

    var K = compute_gram_matrix(X, "linear", gamma=1.0)
    var res = _kernel_svr_solver(
        K, y, C=10.0, epsilon=0.01, fit_intercept=True, tol=1e-4
    )

    assert_true(res.converged)

    # Check fitted values are close to actual targets
    for i in range(4):
        var pred: Scalar[DType.float64] = 0.0
        for m in range(len(res.support_indices)):
            var sv_idx = res.support_indices[m]
            pred += res.dual_coef[m] * K[sv_idx, i]
        pred += res.intercept
        assert_almost_equal(pred, y[i], atol=0.1)


def test_smo_errors() raises:
    var K = Matrix[DType.float64](2, 2, 1.0)
    var y: List[Scalar[DType.float64]] = [1.0, -1.0]

    with assert_raises():
        _ = _smo_binary_svc(K, y, C=-1.0)

    with assert_raises():
        _ = _kernel_svr_solver(K, y, C=1.0, epsilon=-0.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
