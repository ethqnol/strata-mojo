from std.math import sqrt, log, exp, isnan, isinf, abs
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata.core.matrix import Matrix
from strata.utils.random import PRNG
from strata.mixture._gmm_math import (
    LN_2PI,
    _estimate_gaussian_nk_and_means,
    _estimate_covariances_full,
    _estimate_covariances_tied,
    _estimate_covariances_diag,
    _estimate_covariances_spherical,
    _cholesky_lower,
    _precision_cholesky_from_lower,
    _compute_precision_cholesky_full,
    _compute_precision_cholesky_tied,
    _compute_precision_cholesky_diag,
    _compute_precision_cholesky_spherical,
    _compute_precisions_from_cholesky_full,
    _compute_precisions_from_cholesky_tied,
    _compute_precisions_from_cholesky_diag,
    _compute_precisions_from_cholesky_spherical,
    _compute_log_det_cholesky_full,
    _compute_log_det_cholesky_tied,
    _estimate_log_gaussian_prob_full,
    _estimate_log_gaussian_prob_tied,
    _estimate_log_gaussian_prob_diag,
    _estimate_log_gaussian_prob_spherical,
    _estimate_log_prob_resp,
    _prng_uniform_open01,
    _prng_standard_normal_pair,
    _sample_multivariate_normal,
    _sample_categorical_counts,
    _n_parameters,
)


def _matmul(
    A: Matrix[DType.float64], B: Matrix[DType.float64]
) -> Matrix[DType.float64]:
    var C = Matrix[DType.float64](A.rows, B.cols, 0.0)
    for i in range(A.rows):
        for j in range(B.cols):
            var s: Float64 = 0.0
            for m in range(A.cols):
                s += A[i, m] * B[m, j]
            C[i, j] = s
    return C^


def test_cholesky_and_precision_cholesky_identity() raises:
    # 2x2 SPD matrix A2 = [[4, 2], [2, 3]], det = 8
    var A2 = Matrix[DType.float64](2, 2, 0.0)
    A2[0, 0] = 4.0
    A2[0, 1] = 2.0
    A2[1, 0] = 2.0
    A2[1, 1] = 3.0

    var L2 = _cholesky_lower(A2)
    assert_almost_equal(L2[0, 0], 2.0, atol=1e-12)
    assert_almost_equal(L2[0, 1], 0.0, atol=1e-12)
    assert_almost_equal(L2[1, 0], 1.0, atol=1e-12)
    assert_almost_equal(L2[1, 1], sqrt(2.0), atol=1e-12)

    var L2_recon = _matmul(L2, L2.transpose())
    for r in range(2):
        for c in range(2):
            assert_almost_equal(L2_recon[r, c], A2[r, c], atol=1e-12)

    var P2 = _precision_cholesky_from_lower(L2)
    assert_almost_equal(P2[1, 0], 0.0, atol=1e-12)  # Upper triangular
    var prec2 = _compute_precisions_from_cholesky_tied(P2)
    assert_almost_equal(prec2[0, 0], 3.0 / 8.0, atol=1e-12)
    assert_almost_equal(prec2[0, 1], -2.0 / 8.0, atol=1e-12)
    assert_almost_equal(prec2[1, 0], -2.0 / 8.0, atol=1e-12)
    assert_almost_equal(prec2[1, 1], 4.0 / 8.0, atol=1e-12)

    var id2 = _matmul(A2, prec2)
    assert_almost_equal(id2[0, 0], 1.0, atol=1e-12)
    assert_almost_equal(id2[0, 1], 0.0, atol=1e-12)
    assert_almost_equal(id2[1, 0], 0.0, atol=1e-12)
    assert_almost_equal(id2[1, 1], 1.0, atol=1e-12)

    var log_det2 = _compute_log_det_cholesky_tied(P2)
    assert_almost_equal(log_det2, -0.5 * log(8.0), atol=1e-12)

    # 3x3 SPD matrix
    var A3 = Matrix[DType.float64](3, 3, 0.0)
    A3[0, 0] = 4.0
    A3[0, 1] = 2.0
    A3[0, 2] = -2.0
    A3[1, 0] = 2.0
    A3[1, 1] = 10.0
    A3[1, 2] = 2.0
    A3[2, 0] = -2.0
    A3[2, 1] = 2.0
    A3[2, 2] = 14.0

    var L3 = _cholesky_lower(A3)
    var L3_recon = _matmul(L3, L3.transpose())
    for r in range(3):
        for c in range(3):
            assert_almost_equal(L3_recon[r, c], A3[r, c], atol=1e-11)

    var P3 = _precision_cholesky_from_lower(L3)
    for r in range(3):
        for c in range(r):
            assert_almost_equal(P3[r, c], 0.0, atol=1e-12)

    var prec3 = _compute_precisions_from_cholesky_tied(P3)
    var id3 = _matmul(A3, prec3)
    for r in range(3):
        for c in range(3):
            var target = 1.0 if r == c else 0.0
            assert_almost_equal(id3[r, c], target, atol=1e-11)

    # Error handling: non-square, empty, non-SPD
    var non_sq = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = _cholesky_lower(non_sq)

    var empty = Matrix[DType.float64](0, 0, 0.0)
    with assert_raises():
        _ = _cholesky_lower(empty)

    var non_spd = Matrix[DType.float64](2, 2, 0.0)
    non_spd[0, 0] = 1.0
    non_spd[0, 1] = 2.0
    non_spd[1, 0] = 2.0
    non_spd[1, 1] = 1.0
    with assert_raises():
        _ = _cholesky_lower(non_spd)

    var singular = Matrix[DType.float64](2, 2, 0.0)
    singular[0, 0] = 0.0
    singular[1, 1] = 1.0
    with assert_raises():
        _ = _cholesky_lower(singular)


def test_estimate_parameters_and_covariances_all_types() raises:
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 2.0
    X[2, 0] = 4.0
    X[2, 1] = 0.0
    X[3, 0] = 6.0
    X[3, 1] = 4.0

    var resp = Matrix[DType.float64](4, 2, 0.0)
    resp[0, 0] = 0.8
    resp[0, 1] = 0.2
    resp[1, 0] = 0.6
    resp[1, 1] = 0.4
    resp[2, 0] = 0.1
    resp[2, 1] = 0.9
    resp[3, 0] = 0.0
    resp[3, 1] = 1.0

    var stats = _estimate_gaussian_nk_and_means(X, resp)
    assert_almost_equal(stats.nk[0], 1.5, atol=1e-12)
    assert_almost_equal(stats.nk[1], 2.5, atol=1e-12)
    assert_almost_equal(stats.means[0, 0], 1.6 / 1.5, atol=1e-12)
    assert_almost_equal(stats.means[0, 1], 1.2 / 1.5, atol=1e-12)
    assert_almost_equal(stats.means[1, 0], 10.4 / 2.5, atol=1e-12)
    assert_almost_equal(stats.means[1, 1], 4.8 / 2.5, atol=1e-12)

    var reg = 0.01
    var covs_full = _estimate_covariances_full(
        X, resp, stats.nk, stats.means, reg
    )
    var cov_tied = _estimate_covariances_tied(
        X, resp, stats.nk, stats.means, reg
    )
    var cov_diag = _estimate_covariances_diag(
        X, resp, stats.nk, stats.means, reg
    )
    var cov_sph = _estimate_covariances_spherical(
        X, resp, stats.nk, stats.means, reg
    )

    assert_equal(len(covs_full), 2)
    # Verify exact symmetry and diagonal parity between full, diag, spherical
    for k in range(2):
        assert_equal(covs_full[k][0, 1], covs_full[k][1, 0])
        assert_almost_equal(covs_full[k][0, 0], cov_diag[k, 0], atol=1e-12)
        assert_almost_equal(covs_full[k][1, 1], cov_diag[k, 1], atol=1e-12)
        assert_almost_equal(
            cov_sph[k], 0.5 * (cov_diag[k, 0] + cov_diag[k, 1]), atol=1e-12
        )

    assert_equal(cov_tied[0, 1], cov_tied[1, 0])
    # Verify tied equals nk-weighted average of unregularized full covs + reg*I
    var total_nk = stats.nk[0] + stats.nk[1]
    for r in range(2):
        for c in range(2):
            var w0 = stats.nk[0] / total_nk
            var w1 = stats.nk[1] / total_nk
            var unreg0 = covs_full[0][r, c] - (reg if r == c else 0.0)
            var unreg1 = covs_full[1][r, c] - (reg if r == c else 0.0)
            var expected_tied = (
                w0 * unreg0 + w1 * unreg1 + (reg if r == c else 0.0)
            )
            assert_almost_equal(cov_tied[r, c], expected_tied, atol=1e-12)

    # Shift invariance test (+1e6 offset must not suffer catastrophic cancellation)
    var X_shifted = Matrix[DType.float64](4, 2, 0.0)
    for i in range(4):
        for j in range(2):
            X_shifted[i, j] = X[i, j] + 1000000.0

    var stats_s = _estimate_gaussian_nk_and_means(X_shifted, resp)
    var covs_full_s = _estimate_covariances_full(
        X_shifted, resp, stats_s.nk, stats_s.means, reg
    )
    var cov_tied_s = _estimate_covariances_tied(
        X_shifted, resp, stats_s.nk, stats_s.means, reg
    )
    var cov_diag_s = _estimate_covariances_diag(
        X_shifted, resp, stats_s.nk, stats_s.means, reg
    )
    var cov_sph_s = _estimate_covariances_spherical(
        X_shifted, resp, stats_s.nk, stats_s.means, reg
    )

    for k in range(2):
        assert_almost_equal(
            stats_s.means[k, 0], stats.means[k, 0] + 1000000.0, atol=1e-8
        )
        assert_almost_equal(
            stats_s.means[k, 1], stats.means[k, 1] + 1000000.0, atol=1e-8
        )
        assert_almost_equal(cov_sph_s[k], cov_sph[k], atol=1e-8)
        for r in range(2):
            assert_almost_equal(cov_diag_s[k, r], cov_diag[k, r], atol=1e-8)
            for c in range(2):
                assert_almost_equal(
                    covs_full_s[k][r, c], covs_full[k][r, c], atol=1e-8
                )
                assert_almost_equal(cov_tied_s[r, c], cov_tied[r, c], atol=1e-8)


def test_precision_cholesky_all_types_and_invalid_covs() raises:
    var C0 = Matrix[DType.float64](2, 2, 0.0)
    C0[0, 0] = 4.0
    C0[0, 1] = 2.0
    C0[1, 0] = 2.0
    C0[1, 1] = 3.0
    var C1 = Matrix[DType.float64](2, 2, 0.0)
    C1[0, 0] = 2.0
    C1[0, 1] = 0.0
    C1[1, 0] = 0.0
    C1[1, 1] = 5.0
    var covs_full = List[Matrix[DType.float64]]()
    covs_full.append(C0.copy())
    covs_full.append(C1.copy())

    var pchol_full = _compute_precision_cholesky_full(covs_full)
    var precs_full = _compute_precisions_from_cholesky_full(pchol_full)
    var id0 = _matmul(C0, precs_full[0])
    var id1 = _matmul(C1, precs_full[1])
    for r in range(2):
        for c in range(2):
            var target = 1.0 if r == c else 0.0
            assert_almost_equal(id0[r, c], target, atol=1e-12)
            assert_almost_equal(id1[r, c], target, atol=1e-12)

    var pchol_tied = _compute_precision_cholesky_tied(C0)
    var prec_tied = _compute_precisions_from_cholesky_tied(pchol_tied)
    var id_tied = _matmul(C0, prec_tied)
    for r in range(2):
        for c in range(2):
            var target = 1.0 if r == c else 0.0
            assert_almost_equal(id_tied[r, c], target, atol=1e-12)

    var cov_diag = Matrix[DType.float64](2, 2, 0.0)
    cov_diag[0, 0] = 4.0
    cov_diag[0, 1] = 0.25
    cov_diag[1, 0] = 2.0
    cov_diag[1, 1] = 9.0
    var pchol_diag = _compute_precision_cholesky_diag(cov_diag)
    var prec_diag = _compute_precisions_from_cholesky_diag(pchol_diag)
    assert_almost_equal(prec_diag[0, 0], 0.25, atol=1e-12)
    assert_almost_equal(prec_diag[0, 1], 4.0, atol=1e-12)
    assert_almost_equal(prec_diag[1, 0], 0.5, atol=1e-12)
    assert_almost_equal(prec_diag[1, 1], 1.0 / 9.0, atol=1e-12)

    var cov_sph: List[Scalar[DType.float64]] = [4.0, 0.25]
    var pchol_sph = _compute_precision_cholesky_spherical(cov_sph)
    var prec_sph = _compute_precisions_from_cholesky_spherical(pchol_sph)
    assert_almost_equal(prec_sph[0], 0.25, atol=1e-12)
    assert_almost_equal(prec_sph[1], 4.0, atol=1e-12)

    # Reject non-positive covariances
    var bad_diag = cov_diag.copy()
    bad_diag[1, 0] = 0.0
    with assert_raises():
        _ = _compute_precision_cholesky_diag(bad_diag)

    var bad_sph: List[Scalar[DType.float64]] = [1.0, -0.5]
    with assert_raises():
        _ = _compute_precision_cholesky_spherical(bad_sph)


def test_log_gaussian_prob_analytical_parity() raises:
    # 1D exact normal density: mu = 1.5, var = 2.0
    var X1 = Matrix[DType.float64](2, 1, 0.0)
    X1[0, 0] = 1.5
    X1[1, 0] = 3.5
    var means1 = Matrix[DType.float64](1, 1, 1.5)
    var cov1 = Matrix[DType.float64](1, 1, 2.0)
    var pchol1 = _compute_precision_cholesky_diag(cov1)
    var lp1 = _estimate_log_gaussian_prob_diag(X1, means1, pchol1)

    var expected_at_mean = -0.5 * (LN_2PI + log(2.0))
    var expected_at_offset = -0.5 * (LN_2PI + log(2.0) + (2.0 * 2.0) / 2.0)
    assert_almost_equal(lp1[0, 0], expected_at_mean, atol=1e-12)
    assert_almost_equal(lp1[1, 0], expected_at_offset, atol=1e-12)

    # 2D correlated exact normal density: mu = [1, -1], Sigma = [[4, 2], [2, 3]]
    var X2 = Matrix[DType.float64](1, 2, 0.0)
    X2[0, 0] = 3.0
    X2[0, 1] = 0.0
    var means2 = Matrix[DType.float64](1, 2, 0.0)
    means2[0, 0] = 1.0
    means2[0, 1] = -1.0
    var cov2 = Matrix[DType.float64](2, 2, 0.0)
    cov2[0, 0] = 4.0
    cov2[0, 1] = 2.0
    cov2[1, 0] = 2.0
    cov2[1, 1] = 3.0
    var covs2 = List[Matrix[DType.float64]]()
    covs2.append(cov2^)
    var pchol2 = _compute_precision_cholesky_full(covs2)
    var lp2 = _estimate_log_gaussian_prob_full(X2, means2, pchol2)
    # Mahal_sq = 1.0, det = 8.0
    var expected_2d = -0.5 * (2.0 * LN_2PI + log(8.0) + 1.0)
    assert_almost_equal(lp2[0, 0], expected_2d, atol=1e-12)

    # Cross-type equivalence: diagonal covariance -> full == diag
    var X_cross = Matrix[DType.float64](3, 2, 0.0)
    X_cross[0, 0] = -1.0
    X_cross[0, 1] = 2.5
    X_cross[1, 0] = 0.5
    X_cross[1, 1] = -0.5
    X_cross[2, 0] = 3.0
    X_cross[2, 1] = 1.0
    var means_cross = Matrix[DType.float64](2, 2, 0.0)
    means_cross[0, 0] = 0.0
    means_cross[0, 1] = 1.0
    means_cross[1, 0] = 2.0
    means_cross[1, 1] = -1.0

    var dmat = Matrix[DType.float64](2, 2, 0.0)
    dmat[0, 0] = 2.0
    dmat[0, 1] = 3.0
    dmat[1, 0] = 4.0
    dmat[1, 1] = 1.0
    var f_list = List[Matrix[DType.float64]]()
    for k in range(2):
        var m = Matrix[DType.float64](2, 2, 0.0)
        m[0, 0] = dmat[k, 0]
        m[1, 1] = dmat[k, 1]
        f_list.append(m^)

    var lp_diag = _estimate_log_gaussian_prob_diag(
        X_cross, means_cross, _compute_precision_cholesky_diag(dmat)
    )
    var lp_full = _estimate_log_gaussian_prob_full(
        X_cross, means_cross, _compute_precision_cholesky_full(f_list)
    )
    for i in range(3):
        for k in range(2):
            assert_almost_equal(lp_full[i, k], lp_diag[i, k], atol=1e-12)

    # Cross-type equivalence: shared isotropic 2.0 * I -> full == tied == diag == spherical
    var iso_tied = Matrix[DType.float64](2, 2, 0.0)
    iso_tied[0, 0] = 2.0
    iso_tied[1, 1] = 2.0
    var iso_full = List[Matrix[DType.float64]]()
    iso_full.append(iso_tied.copy())
    iso_full.append(iso_tied.copy())
    var iso_diag = Matrix[DType.float64](2, 2, 2.0)
    var iso_sph: List[Scalar[DType.float64]] = [2.0, 2.0]

    var lp_iso_full = _estimate_log_gaussian_prob_full(
        X_cross, means_cross, _compute_precision_cholesky_full(iso_full)
    )
    var lp_iso_tied = _estimate_log_gaussian_prob_tied(
        X_cross, means_cross, _compute_precision_cholesky_tied(iso_tied)
    )
    var lp_iso_diag = _estimate_log_gaussian_prob_diag(
        X_cross, means_cross, _compute_precision_cholesky_diag(iso_diag)
    )
    var lp_iso_sph = _estimate_log_gaussian_prob_spherical(
        X_cross, means_cross, _compute_precision_cholesky_spherical(iso_sph)
    )
    for i in range(3):
        for k in range(2):
            assert_almost_equal(
                lp_iso_full[i, k], lp_iso_tied[i, k], atol=1e-12
            )
            assert_almost_equal(
                lp_iso_full[i, k], lp_iso_diag[i, k], atol=1e-12
            )
            assert_almost_equal(lp_iso_full[i, k], lp_iso_sph[i, k], atol=1e-12)


def test_estep_log_prob_resp_normalization_and_extreme_values() raises:
    var lp = Matrix[DType.float64](2, 2, 0.0)
    lp[0, 0] = log(0.5)
    lp[0, 1] = log(0.25)
    lp[1, 0] = 0.0
    lp[1, 1] = 0.0
    var weights: List[Scalar[DType.float64]] = [0.2, 0.8]

    var res = _estimate_log_prob_resp(lp, weights)
    assert_almost_equal(res.log_prob_norm[0], log(0.3), atol=1e-9)
    assert_almost_equal(res.log_prob_norm[1], 0.0, atol=1e-9)
    assert_almost_equal(res.mean_log_prob, 0.5 * log(0.3), atol=1e-9)

    assert_almost_equal(exp(res.log_resp[0, 0]), 1.0 / 3.0, atol=1e-9)
    assert_almost_equal(exp(res.log_resp[0, 1]), 2.0 / 3.0, atol=1e-9)
    assert_almost_equal(exp(res.log_resp[1, 0]), 0.2, atol=1e-9)
    assert_almost_equal(exp(res.log_resp[1, 1]), 0.8, atol=1e-9)

    # Extreme log probabilities must remain finite and sum to 1
    var lp_ext = Matrix[DType.float64](2, 2, 0.0)
    lp_ext[0, 0] = -10000.0
    lp_ext[0, 1] = -9999.0
    lp_ext[1, 0] = 500.0
    lp_ext[1, 1] = 498.0
    var uniform_w: List[Scalar[DType.float64]] = [0.5, 0.5]
    var res_ext = _estimate_log_prob_resp(lp_ext, uniform_w)
    for i in range(2):
        assert_false(isnan(res_ext.log_prob_norm[i]))
        assert_false(isinf(res_ext.log_prob_norm[i]))
        var row_sum = exp(res_ext.log_resp[i, 0]) + exp(res_ext.log_resp[i, 1])
        assert_almost_equal(row_sum, 1.0, atol=1e-9)

    # Zero-weight component should cleanly receive zero posterior probability
    var zero_w: List[Scalar[DType.float64]] = [0.0, 1.0]
    var res_zw = _estimate_log_prob_resp(lp, zero_w)
    assert_almost_equal(exp(res_zw.log_resp[0, 0]), 0.0, atol=1e-12)
    assert_almost_equal(exp(res_zw.log_resp[0, 1]), 1.0, atol=1e-12)

    # All-zero weights must raise
    var bad_w: List[Scalar[DType.float64]] = [0.0, 0.0]
    with assert_raises():
        _ = _estimate_log_prob_resp(lp, bad_w)


def test_sampling_helpers_and_n_parameters() raises:
    var rng = PRNG(12345)
    for _ in range(200):
        var u = _prng_uniform_open01(rng)
        assert_true(u > 0.0 and u < 1.0)

    # Box-Muller mean & variance check + determinism
    var rng_a = PRNG(77)
    var rng_b = PRNG(77)
    var p_a = _prng_standard_normal_pair(rng_a)
    var p_b = _prng_standard_normal_pair(rng_b)
    assert_equal(p_a[0], p_b[0])
    assert_equal(p_a[1], p_b[1])

    var rng_z = PRNG(2025)
    var sum_z: Float64 = 0.0
    var sum_z2: Float64 = 0.0
    var n_z = 2000
    for _ in range(n_z // 2):
        var pair = _prng_standard_normal_pair(rng_z)
        sum_z += pair[0] + pair[1]
        sum_z2 += pair[0] * pair[0] + pair[1] * pair[1]
    var mean_z = sum_z / Float64(n_z)
    var var_z = sum_z2 / Float64(n_z) - mean_z * mean_z
    assert_almost_equal(mean_z, 0.0, atol=0.1)
    assert_almost_equal(var_z, 1.0, atol=0.15)

    # Correlated 2D multivariate normal sampling check
    var mu: List[Scalar[DType.float64]] = [2.0, -3.0]
    var cov = Matrix[DType.float64](2, 2, 0.0)
    cov[0, 0] = 4.0
    cov[0, 1] = 2.0
    cov[1, 0] = 2.0
    cov[1, 1] = 3.0
    var L = _cholesky_lower(cov)

    var n_mvn = 2000
    var samples = Matrix[DType.float64](n_mvn, 2, 0.0)
    var rng_mvn = PRNG(99)
    var s0: Float64 = 0.0
    var s1: Float64 = 0.0
    for i in range(n_mvn):
        var v = _sample_multivariate_normal(rng_mvn, mu, L)
        samples[i, 0] = v[0]
        samples[i, 1] = v[1]
        s0 += v[0]
        s1 += v[1]
    var m0 = s0 / Float64(n_mvn)
    var m1 = s1 / Float64(n_mvn)
    assert_almost_equal(m0, 2.0, atol=0.15)
    assert_almost_equal(m1, -3.0, atol=0.15)

    var c00: Float64 = 0.0
    var c01: Float64 = 0.0
    var c11: Float64 = 0.0
    for i in range(n_mvn):
        var d0 = samples[i, 0] - m0
        var d1 = samples[i, 1] - m1
        c00 += d0 * d0
        c01 += d0 * d1
        c11 += d1 * d1
    assert_almost_equal(c00 / Float64(n_mvn), 4.0, atol=0.35)
    assert_almost_equal(c01 / Float64(n_mvn), 2.0, atol=0.35)
    assert_almost_equal(c11 / Float64(n_mvn), 3.0, atol=0.35)

    # Categorical counts
    var rng_cat = PRNG(42)
    var cat_w: List[Float64] = [0.2, 0.8]
    var counts = _sample_categorical_counts(rng_cat, cat_w, 1000)
    assert_equal(len(counts), 2)
    assert_equal(counts[0] + counts[1], 1000)
    assert_true(counts[1] > counts[0])
    assert_almost_equal(Float64(counts[0]) / 1000.0, 0.2, atol=0.05)

    # _n_parameters across all 4 covariance types
    assert_equal(_n_parameters(3, 4, "full"), 44)
    assert_equal(_n_parameters(3, 4, "tied"), 24)
    assert_equal(_n_parameters(3, 4, "diag"), 26)
    assert_equal(_n_parameters(3, 4, "spherical"), 17)
    with assert_raises():
        _ = _n_parameters(3, 4, "invalid")


def test_gmm_math_float32_support() raises:
    var X = Matrix[DType.float32](4, 2, 0.0)
    X[0, 0] = -2.0
    X[0, 1] = -1.0
    X[1, 0] = -1.5
    X[1, 1] = -2.0
    X[2, 0] = 2.0
    X[2, 1] = 1.5
    X[3, 0] = 1.5
    X[3, 1] = 2.0

    var resp = Matrix[DType.float32](4, 2, 0.0)
    resp[0, 0] = 0.9
    resp[0, 1] = 0.1
    resp[1, 0] = 0.8
    resp[1, 1] = 0.2
    resp[2, 0] = 0.1
    resp[2, 1] = 0.9
    resp[3, 0] = 0.2
    resp[3, 1] = 0.8

    var stats = _estimate_gaussian_nk_and_means(X, resp)
    var weights: List[Scalar[DType.float32]] = [0.5, 0.5]

    var c_full = _estimate_covariances_full(
        X, resp, stats.nk, stats.means, 1e-4
    )
    var c_tied = _estimate_covariances_tied(
        X, resp, stats.nk, stats.means, 1e-4
    )
    var c_diag = _estimate_covariances_diag(
        X, resp, stats.nk, stats.means, 1e-4
    )
    var c_sph = _estimate_covariances_spherical(
        X, resp, stats.nk, stats.means, 1e-4
    )

    var lp_full = _estimate_log_gaussian_prob_full(
        X, stats.means, _compute_precision_cholesky_full(c_full)
    )
    var lp_tied = _estimate_log_gaussian_prob_tied(
        X, stats.means, _compute_precision_cholesky_tied(c_tied)
    )
    var lp_diag = _estimate_log_gaussian_prob_diag(
        X, stats.means, _compute_precision_cholesky_diag(c_diag)
    )
    var lp_sph = _estimate_log_gaussian_prob_spherical(
        X, stats.means, _compute_precision_cholesky_spherical(c_sph)
    )

    var res_full = _estimate_log_prob_resp(lp_full, weights)
    var res_tied = _estimate_log_prob_resp(lp_tied, weights)
    var res_diag = _estimate_log_prob_resp(lp_diag, weights)
    var res_sph = _estimate_log_prob_resp(lp_sph, weights)

    for i in range(4):
        assert_false(isnan(Float64(res_full.log_prob_norm[i])))
        assert_false(isnan(Float64(res_tied.log_prob_norm[i])))
        assert_false(isnan(Float64(res_diag.log_prob_norm[i])))
        assert_false(isnan(Float64(res_sph.log_prob_norm[i])))
        var sum_f = exp(Float64(res_full.log_resp[i, 0])) + exp(
            Float64(res_full.log_resp[i, 1])
        )
        assert_almost_equal(sum_f, 1.0, atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
