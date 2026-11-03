from std.math import abs, exp, tanh
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    assert_almost_equal,
)
from strata.core.matrix import Matrix
from strata.svm._kernels import (
    compute_gamma,
    kernel_eval_rows,
    kernel_eval_vec_row,
    compute_kernel_matrix,
    compute_gram_matrix,
)
from strata.exceptions.errors import (
    InvalidParameterError,
    DimensionMismatchError,
)


def test_compute_gamma() raises:
    # 2 samples, 2 features
    var X = Matrix[DType.float64](2, 2, 0.0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 3.0
    X[1, 1] = 4.0

    # auto: 1.0 / n_features = 1.0 / 2 = 0.5
    var g_auto = compute_gamma[DType.float64](X, gamma_str="auto")
    assert_almost_equal(g_auto, 0.5, atol=1e-5)

    # explicit numeric value
    var g_val = compute_gamma[DType.float64](X, gamma_val=0.25)
    assert_almost_equal(g_val, 0.25, atol=1e-5)

    # scale: mean = 2.5, var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
    # gamma = 1 / (2 * 1.25) = 1 / 2.5 = 0.4
    var g_scale = compute_gamma[DType.float64](X, gamma_str="scale")
    assert_almost_equal(g_scale, 0.4, atol=1e-5)

    # invalid gamma string
    with assert_raises():
        _ = compute_gamma[DType.float64](X, gamma_str="invalid_gamma")


def test_linear_kernel() raises:
    var X1 = Matrix[DType.float64](2, 3, 0.0)
    X1[0, 0] = 1.0
    X1[0, 1] = 2.0
    X1[0, 2] = 3.0
    X1[1, 0] = 4.0
    X1[1, 1] = 5.0
    X1[1, 2] = 6.0

    var X2 = Matrix[DType.float64](2, 3, 0.0)
    X2[0, 0] = 0.5
    X2[0, 1] = 1.0
    X2[0, 2] = 1.5
    X2[1, 0] = 2.0
    X2[1, 1] = 0.0
    X2[1, 2] = 1.0

    # X1[0] . X2[0] = 1*0.5 + 2*1.0 + 3*1.5 = 0.5 + 2.0 + 4.5 = 7.0
    var k00 = kernel_eval_rows(X1, 0, X2, 0, "linear", gamma=1.0)
    assert_almost_equal(k00, 7.0, atol=1e-5)

    # X1[0] . X2[1] = 1*2.0 + 2*0.0 + 3*1.0 = 5.0
    var k01 = kernel_eval_rows(X1, 0, X2, 1, "linear", gamma=1.0)
    assert_almost_equal(k01, 5.0, atol=1e-5)

    var K = compute_kernel_matrix(X1, X2, "linear", gamma=1.0)
    assert_equal(K.rows, 2)
    assert_equal(K.cols, 2)
    assert_almost_equal(K[0, 0], 7.0, atol=1e-5)
    assert_almost_equal(K[0, 1], 5.0, atol=1e-5)


def test_rbf_kernel() raises:
    var X = Matrix[DType.float64](3, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 1.0
    X[1, 1] = 1.0
    X[2, 0] = 2.0
    X[2, 1] = 2.0

    var gamma = 0.5
    # K(X[0], X[0]) = exp(0) = 1.0
    var k00 = kernel_eval_rows(X, 0, X, 0, "rbf", gamma=gamma)
    assert_almost_equal(k00, 1.0, atol=1e-7)

    # dist^2(X[0], X[1]) = 1^2 + 1^2 = 2.0 -> exp(-0.5 * 2.0) = exp(-1.0) = 0.36787944117
    var k01 = kernel_eval_rows(X, 0, X, 1, "rbf", gamma=gamma)
    var expected_k01 = exp(-1.0)
    assert_almost_equal(k01, expected_k01, atol=1e-5)

    # Gram matrix
    var G = compute_gram_matrix(X, "rbf", gamma=gamma)
    assert_equal(G.rows, 3)
    assert_equal(G.cols, 3)
    for i in range(3):
        assert_almost_equal(G[i, i], 1.0, atol=1e-7)
    assert_almost_equal(G[0, 1], expected_k01, atol=1e-5)
    assert_almost_equal(G[1, 0], expected_k01, atol=1e-5)

    # Compare with compute_kernel_matrix
    var K_full = compute_kernel_matrix(X, X, "rbf", gamma=gamma)
    for r in range(3):
        for c in range(3):
            assert_almost_equal(G[r, c], K_full[r, c], atol=1e-7)


def test_poly_and_sigmoid_kernels() raises:
    var X = Matrix[DType.float64](2, 2, 0.0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 0.5
    X[1, 1] = 1.5

    # dot(X[0], X[1]) = 1*0.5 + 2*1.5 = 3.5
    # Poly degree 2, gamma 0.5, coef0 1.0: (0.5 * 3.5 + 1.0)^2 = (1.75 + 1.0)^2 = 2.75^2 = 7.5625
    var k_poly = kernel_eval_rows(
        X, 0, X, 1, "poly", gamma=0.5, degree=2, coef0=1.0
    )
    assert_almost_equal(k_poly, 7.5625, atol=1e-5)

    # Sigmoid gamma 0.5, coef0 -1.0: tanh(0.5 * 3.5 - 1.0) = tanh(0.75)
    var k_sig = kernel_eval_rows(X, 0, X, 1, "sigmoid", gamma=0.5, coef0=-1.0)
    var expected_sig = tanh(0.75)
    assert_almost_equal(k_sig, expected_sig, atol=1e-5)


def test_kernel_eval_vec_row() raises:
    var X = Matrix[DType.float64](2, 3, 0.0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[0, 2] = 3.0
    X[1, 0] = 4.0
    X[1, 1] = 5.0
    X[1, 2] = 6.0

    var v: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]

    # v matches X[0], so RBF kernel should be 1.0
    var k_rbf = kernel_eval_vec_row(v, X, 0, "rbf", gamma=0.5)
    assert_almost_equal(k_rbf, 1.0, atol=1e-7)

    # linear kernel with row 1: 1*4 + 2*5 + 3*6 = 32.0
    var k_lin = kernel_eval_vec_row(v, X, 1, "linear", gamma=1.0)
    assert_almost_equal(k_lin, 32.0, atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
