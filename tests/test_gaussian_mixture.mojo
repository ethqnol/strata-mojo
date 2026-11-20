from std.math import log, exp, abs, isnan, isinf, nan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import Matrix, Dataset, GaussianMixture, GaussianSample
from strata.base import fit, predict
from strata.io.serializer import dumps, loads
from strata.mixture._gmm_math import _n_parameters, _prng_standard_normal_pair
from strata.utils.random import PRNG


def _make_three_clusters_2d() -> Matrix[DType.float64]:
    # 18 points across 3 well-separated 2D clusters:
    # Cluster A around (0, 0), Cluster B around (10, 0), Cluster C around (0, 10)
    var X = Matrix[DType.float64](18, 2, 0.0)
    var offsets: List[Float64] = [
        -0.4,
        -0.2,
        0.3,
        -0.3,
        -0.2,
        0.4,
        0.4,
        0.2,
        0.1,
        -0.4,
        -0.2,
        0.3,
    ]
    for i in range(6):
        X[i, 0] = offsets[2 * i]
        X[i, 1] = offsets[2 * i + 1]
        X[6 + i, 0] = 10.0 + offsets[2 * i]
        X[6 + i, 1] = offsets[2 * i + 1]
        X[12 + i, 0] = offsets[2 * i]
        X[12 + i, 1] = 10.0 + offsets[2 * i + 1]
    return X^


def _make_two_clusters_2d() -> Matrix[DType.float64]:
    var X = Matrix[DType.float64](12, 2, 0.0)
    var offsets: List[Float64] = [
        -0.3,
        -0.2,
        0.2,
        -0.3,
        -0.2,
        0.3,
        0.3,
        0.2,
        0.1,
        -0.1,
        -0.1,
        0.1,
    ]
    for i in range(6):
        X[i, 0] = -6.0 + offsets[2 * i]
        X[i, 1] = -6.0 + offsets[2 * i + 1]
        X[6 + i, 0] = 6.0 + offsets[2 * i]
        X[6 + i, 1] = 6.0 + offsets[2 * i + 1]
    return X^


def test_gmm_well_separated_clusters_all_cov_types() raises:
    var X = _make_three_clusters_2d()
    var cov_types: List[String] = ["full", "tied", "diag", "spherical"]

    for t in range(len(cov_types)):
        var ctype = cov_types[t]
        var gmm = GaussianMixture[DType.float64](
            n_components=3, covariance_type=ctype, tol=1e-4, random_state=42
        )
        gmm.fit(X)

        assert_true(gmm.is_fitted)
        assert_true(gmm.converged_)
        assert_equal(gmm.n_features_in_, 2)
        assert_equal(len(gmm.weights_), 3)
        assert_equal(gmm.means_.rows, 3)
        assert_equal(gmm.means_.cols, 2)

        if ctype == "full":
            assert_equal(len(gmm.covariances_), 3)
            assert_equal(gmm.covariances_[0].rows, 2)
            assert_equal(gmm.covariances_[0].cols, 2)
            assert_equal(len(gmm.precisions_), 3)
            assert_equal(len(gmm.precisions_cholesky_), 3)
        elif ctype == "tied":
            assert_equal(len(gmm.covariances_), 1)
            assert_equal(gmm.covariances_[0].rows, 2)
            assert_equal(gmm.covariances_[0].cols, 2)
        elif ctype == "diag":
            assert_equal(len(gmm.covariances_), 1)
            assert_equal(gmm.covariances_[0].rows, 3)
            assert_equal(gmm.covariances_[0].cols, 2)
        else:
            assert_equal(len(gmm.covariances_), 1)
            assert_equal(gmm.covariances_[0].rows, 3)
            assert_equal(gmm.covariances_[0].cols, 1)

        var w_sum: Float64 = 0.0
        for k in range(3):
            w_sum += gmm.weights_[k]
            assert_almost_equal(gmm.weights_[k], 1.0 / 3.0, atol=0.05)
        assert_almost_equal(w_sum, 1.0, atol=1e-9)

        var preds = gmm.predict(X)
        assert_equal(len(preds), 18)
        var l0 = preds[0]
        var l1 = preds[6]
        var l2 = preds[12]
        assert_true(l0 != l1 and l1 != l2 and l0 != l2)
        for i in range(6):
            assert_equal(preds[i], l0)
            assert_equal(preds[6 + i], l1)
            assert_equal(preds[12 + i], l2)

        var probs = gmm.predict_proba(X)
        assert_equal(probs.rows, 18)
        assert_equal(probs.cols, 3)
        for i in range(18):
            var rsum = probs[i, 0] + probs[i, 1] + probs[i, 2]
            assert_almost_equal(rsum, 1.0, atol=1e-9)
            assert_true(probs[i, preds[i]] > 0.99)

        var gmm_fp = GaussianMixture[DType.float64](
            n_components=3, covariance_type=ctype, tol=1e-4, random_state=42
        )
        var fp_preds = gmm_fp.fit_predict(X)
        for i in range(18):
            assert_equal(fp_preds[i], preds[i])


def test_gmm_anisotropic_correlated_full_covariance() raises:
    # Two tilted correlated clusters:
    # Cluster 0 near (0, 0) stretched along +45 deg (positive correlation)
    # Cluster 1 near (6, -6) stretched along -45 deg (negative correlation)
    var X = Matrix[DType.float64](20, 2, 0.0)
    for i in range(10):
        var t = Float64(i) - 4.5
        var perp = 0.15 if (i % 2 == 0) else -0.15
        X[i, 0] = t + perp
        X[i, 1] = t - perp
        X[10 + i, 0] = 6.0 + t + perp
        X[10 + i, 1] = -6.0 - t + perp

    var gmm_full = GaussianMixture[DType.float64](
        n_components=2, covariance_type="full", n_init=3, random_state=42
    )
    gmm_full.fit(X)

    var gmm_sph = GaussianMixture[DType.float64](
        n_components=2, covariance_type="spherical", n_init=3, random_state=42
    )
    gmm_sph.fit(X)

    # Full covariance must achieve higher log-likelihood than isotropic spherical
    assert_true(gmm_full.score(X) > gmm_sph.score(X))
    # One component must have positive off-diagonal cov, the other negative
    var c01_a = gmm_full.covariances_[0][0, 1]
    var c01_b = gmm_full.covariances_[1][0, 1]
    assert_true(c01_a * c01_b < 0.0)


def test_gmm_single_component_exact_mle() raises:
    var X = Matrix[DType.float64](5, 2, 0.0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 3.0
    X[1, 1] = 4.0
    X[2, 0] = 5.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 6.0
    X[4, 0] = 4.0
    X[4, 1] = 6.0

    # Empirical mean = [3.0, 4.0]
    var exp_c00: Float64 = 0.0
    var exp_c01: Float64 = 0.0
    var exp_c11: Float64 = 0.0
    for i in range(5):
        var d0 = X[i, 0] - 3.0
        var d1 = X[i, 1] - 4.0
        exp_c00 += d0 * d0
        exp_c01 += d0 * d1
        exp_c11 += d1 * d1
    exp_c00 /= 5.0
    exp_c01 /= 5.0
    exp_c11 /= 5.0

    var reg = 1e-6
    var cov_types: List[String] = ["full", "tied", "diag", "spherical"]
    for t in range(len(cov_types)):
        var ctype = cov_types[t]
        var gmm = GaussianMixture[DType.float64](
            n_components=1, covariance_type=ctype, reg_covar=reg
        )
        gmm.fit(X)
        assert_almost_equal(gmm.weights_[0], 1.0, atol=1e-9)
        assert_almost_equal(gmm.means_[0, 0], 3.0, atol=1e-9)
        assert_almost_equal(gmm.means_[0, 1], 4.0, atol=1e-9)

        if ctype == "full" or ctype == "tied":
            assert_almost_equal(
                gmm.covariances_[0][0, 0], exp_c00 + reg, atol=1e-9
            )
            assert_almost_equal(gmm.covariances_[0][0, 1], exp_c01, atol=1e-9)
            assert_almost_equal(
                gmm.covariances_[0][1, 1], exp_c11 + reg, atol=1e-9
            )
            # Verify cov * prec = I
            var p00 = gmm.precisions_[0][0, 0]
            var p01 = gmm.precisions_[0][0, 1]
            var p11 = gmm.precisions_[0][1, 1]
            var c00 = gmm.covariances_[0][0, 0]
            var c01 = gmm.covariances_[0][0, 1]
            var c11 = gmm.covariances_[0][1, 1]
            assert_almost_equal(c00 * p00 + c01 * p01, 1.0, atol=1e-7)
            assert_almost_equal(c00 * p01 + c01 * p11, 0.0, atol=1e-7)
            assert_almost_equal(c01 * p01 + c11 * p11, 1.0, atol=1e-7)
        elif ctype == "diag":
            assert_almost_equal(
                gmm.covariances_[0][0, 0], exp_c00 + reg, atol=1e-9
            )
            assert_almost_equal(
                gmm.covariances_[0][0, 1], exp_c11 + reg, atol=1e-9
            )
        else:
            var exp_sph = 0.5 * (exp_c00 + exp_c11) + reg
            assert_almost_equal(gmm.covariances_[0][0, 0], exp_sph, atol=1e-9)


def test_gmm_all_init_params_and_n_init() raises:
    var X = _make_three_clusters_2d()
    var inits: List[String] = [
        "kmeans",
        "k-means++",
        "random",
        "random_from_data",
    ]

    for i in range(len(inits)):
        var strat = inits[i]
        var gmm = GaussianMixture[DType.float64](
            n_components=3,
            init_params=strat,
            n_init=3,
            max_iter=200,
            random_state=17,
        )
        gmm.fit(X)
        assert_true(gmm.is_fitted)
        assert_true(gmm.converged_)
        assert_false(isnan(gmm.lower_bound_))

    # Determinism with identical random_state
    var gmm_a = GaussianMixture[DType.float64](
        n_components=3, init_params="random", n_init=2, random_state=99
    )
    var gmm_b = GaussianMixture[DType.float64](
        n_components=3, init_params="random", n_init=2, random_state=99
    )
    gmm_a.fit(X)
    gmm_b.fit(X)
    assert_almost_equal(gmm_a.lower_bound_, gmm_b.lower_bound_, atol=1e-12)
    var pa = gmm_a.predict(X)
    var pb = gmm_b.predict(X)
    for i in range(18):
        assert_equal(pa[i], pb[i])


def test_gmm_custom_initializations() raises:
    var X = _make_two_clusters_2d()
    var w_init: List[Scalar[DType.float64]] = [0.5, 0.5]
    var m_init = Matrix[DType.float64](2, 2, 0.0)
    m_init[0, 0] = -6.0
    m_init[0, 1] = -6.0
    m_init[1, 0] = 6.0
    m_init[1, 1] = 6.0

    # Full precisions_init (2 matrices of 2x2)
    var p_full = List[Matrix[DType.float64]]()
    var p_mat = Matrix[DType.float64](2, 2, 0.0)
    p_mat[0, 0] = 2.0
    p_mat[0, 1] = 0.5
    p_mat[1, 0] = 0.5
    p_mat[1, 1] = 2.0
    p_full.append(p_mat.copy())
    p_full.append(p_mat.copy())

    var gmm_full = GaussianMixture[DType.float64](
        n_components=2,
        covariance_type="full",
        weights_init=w_init,
        means_init=m_init,
        precisions_init=p_full,
    )
    gmm_full.fit(X)
    assert_true(gmm_full.converged_)
    var preds_f = gmm_full.predict(X)
    for i in range(6):
        assert_equal(preds_f[i], 0)
        assert_equal(preds_f[6 + i], 1)

    # Tied precisions_init (1 matrix of 2x2)
    var p_tied = List[Matrix[DType.float64]]()
    p_tied.append(p_mat.copy())
    var gmm_tied = GaussianMixture[DType.float64](
        n_components=2,
        covariance_type="tied",
        weights_init=w_init,
        means_init=m_init,
        precisions_init=p_tied,
    )
    gmm_tied.fit(X)
    assert_true(gmm_tied.converged_)

    # Diag precisions_init (1 matrix of 2x2)
    var p_diag_mat = Matrix[DType.float64](2, 2, 2.0)
    var p_diag = List[Matrix[DType.float64]]()
    p_diag.append(p_diag_mat^)
    var gmm_diag = GaussianMixture[DType.float64](
        n_components=2,
        covariance_type="diag",
        weights_init=w_init,
        means_init=m_init,
        precisions_init=p_diag,
    )
    gmm_diag.fit(X)
    assert_true(gmm_diag.converged_)

    # Spherical precisions_init (1 matrix of 2x1)
    var p_sph_mat = Matrix[DType.float64](2, 1, 2.0)
    var p_sph = List[Matrix[DType.float64]]()
    p_sph.append(p_sph_mat^)
    var gmm_sph = GaussianMixture[DType.float64](
        n_components=2,
        covariance_type="spherical",
        weights_init=w_init,
        means_init=m_init,
        precisions_init=p_sph,
    )
    gmm_sph.fit(X)
    assert_true(gmm_sph.converged_)


def test_gmm_score_samples_aic_and_bic_model_selection() raises:
    var X = _make_three_clusters_2d()
    var gmm3 = GaussianMixture[DType.float64](
        n_components=3, covariance_type="full", random_state=42
    )
    gmm3.fit(X)

    var ll_samples = gmm3.score_samples(X)
    assert_equal(len(ll_samples), 18)
    var mean_ll: Float64 = 0.0
    for i in range(18):
        mean_ll += ll_samples[i]
    mean_ll /= 18.0

    var avg_score = gmm3.score(X)
    assert_almost_equal(avg_score, mean_ll, atol=1e-9)

    var n_params = Float64(_n_parameters(3, 2, "full"))
    var expected_aic = -2.0 * avg_score * 18.0 + 2.0 * n_params
    var expected_bic = -2.0 * avg_score * 18.0 + n_params * log(18.0)
    assert_almost_equal(gmm3.aic(X), expected_aic, atol=1e-8)
    assert_almost_equal(gmm3.bic(X), expected_bic, atol=1e-8)

    # BIC model selection on 150 Gaussian samples from 3 clusters:
    # true K=3 must beat underfit K=1 and overfit K=6
    var X_bic = Matrix[DType.float64](150, 2, 0.0)
    var rng_bic = PRNG(123)
    var centers_x: List[Float64] = [0.0, 10.0, 0.0]
    var centers_y: List[Float64] = [0.0, 0.0, 10.0]
    for c in range(3):
        for i in range(50):
            var z = _prng_standard_normal_pair(rng_bic)
            var row = c * 50 + i
            X_bic[row, 0] = centers_x[c] + 0.8 * z[0]
            X_bic[row, 1] = centers_y[c] + 0.8 * z[1]

    var gmm_bic_3 = GaussianMixture[DType.float64](
        n_components=3, covariance_type="full", random_state=42
    )
    gmm_bic_3.fit(X_bic)
    var gmm_bic_1 = GaussianMixture[DType.float64](
        n_components=1, covariance_type="full", random_state=42
    )
    gmm_bic_1.fit(X_bic)
    var gmm_bic_6 = GaussianMixture[DType.float64](
        n_components=6, covariance_type="full", random_state=42
    )
    gmm_bic_6.fit(X_bic)
    assert_true(gmm_bic_3.bic(X_bic) < gmm_bic_1.bic(X_bic))
    assert_true(gmm_bic_3.bic(X_bic) < gmm_bic_6.bic(X_bic))


def test_gmm_sample_generation_all_cov_types() raises:
    var X = _make_two_clusters_2d()
    var cov_types: List[String] = ["full", "tied", "diag", "spherical"]

    for t in range(len(cov_types)):
        var ctype = cov_types[t]
        var gmm = GaussianMixture[DType.float64](
            n_components=2, covariance_type=ctype, random_state=42
        )
        gmm.fit(X)

        var samp_a = gmm.sample(n_samples=300, random_state=123)
        var samp_b = gmm.sample(n_samples=300, random_state=123)
        assert_equal(samp_a.X.rows, 300)
        assert_equal(samp_a.X.cols, 2)
        assert_equal(len(samp_a.y), 300)

        # Deterministic seeding
        for i in range(300):
            assert_equal(samp_a.y[i], samp_b.y[i])
            assert_equal(samp_a.X[i, 0], samp_b.X[i, 0])
            assert_equal(samp_a.X[i, 1], samp_b.X[i, 1])

        var pred_y = gmm.predict(samp_a.X)
        var matches = 0
        for i in range(300):
            assert_true(samp_a.y[i] == 0 or samp_a.y[i] == 1)
            if pred_y[i] == samp_a.y[i]:
                matches += 1
        assert_true(Float64(matches) / 300.0 > 0.90)


def test_gmm_warm_start() raises:
    var X = _make_three_clusters_2d()
    var gmm = GaussianMixture[DType.float64](
        n_components=3,
        init_params="random",
        max_iter=1,
        tol=1e-6,
        warm_start=True,
        random_state=42,
    )
    gmm.fit(X)
    var lb1 = gmm.lower_bound_
    assert_false(gmm.converged_)

    gmm.max_iter = 100
    gmm.fit(X)
    assert_true(gmm.converged_)
    assert_true(gmm.lower_bound_ >= lb1 - 1e-10)

    # Mutating n_components or covariance_type after fitting with warm_start must raise
    gmm.n_components = 2
    with assert_raises():
        gmm.fit(X)
    gmm.n_components = 3
    gmm.covariance_type = "diag"
    with assert_raises():
        gmm.fit(X)
    gmm.covariance_type = "full"


def test_gmm_dataset_clusterer_trait_integration() raises:
    var X = _make_two_clusters_2d()
    var y = List[Scalar[DType.int32]](capacity=12)
    for i in range(12):
        y.append(Scalar[DType.int32](0 if i < 6 else 1))
    var ds = Dataset[DType.float64, DType.int32](X^, y^)

    var gmm = GaussianMixture[DType.float64](
        n_components=2, covariance_type="diag", random_state=42
    )
    fit(gmm, ds)
    assert_true(gmm.is_fitted)

    var preds = predict(gmm, ds)
    assert_equal(len(preds), 12)
    var c0 = preds[0]
    var c1 = preds[6]
    assert_true(c0 != c1)
    for i in range(6):
        assert_equal(preds[i], c0)
        assert_equal(preds[6 + i], c1)


def test_gmm_serialization_all_cov_types_and_unfitted() raises:
    # Unfitted with custom initializations
    var w_init: List[Scalar[DType.float64]] = [0.4, 0.6]
    var m_init = Matrix[DType.float64](2, 2, 1.0)
    var p_init_mat = Matrix[DType.float64](2, 2, 1.0)
    var p_init = List[Matrix[DType.float64]]()
    p_init.append(p_init_mat^)
    var unfitted = GaussianMixture[DType.float64](
        n_components=2,
        covariance_type="diag",
        tol=1e-4,
        reg_covar=1e-5,
        max_iter=50,
        n_init=2,
        init_params="random",
        weights_init=w_init,
        means_init=m_init,
        precisions_init=p_init,
        random_state=7,
        warm_start=True,
    )
    var u_bytes = dumps(unfitted)
    var u_restored = loads[GaussianMixture[DType.float64]](u_bytes)
    assert_false(u_restored.is_fitted)
    assert_equal(u_restored.n_components, 2)
    assert_equal(u_restored.covariance_type, "diag")
    assert_almost_equal(u_restored.tol, 1e-4, atol=1e-12)
    assert_almost_equal(u_restored.reg_covar, 1e-5, atol=1e-12)
    assert_equal(u_restored.max_iter, 50)
    assert_equal(u_restored.n_init, 2)
    assert_equal(u_restored.init_params, "random")
    assert_equal(len(u_restored.weights_init), 2)
    assert_equal(u_restored.means_init.rows, 2)
    assert_equal(len(u_restored.precisions_init), 1)
    assert_true(u_restored.warm_start)

    # Fitted across all 4 covariance types
    var X = _make_three_clusters_2d()
    var cov_types: List[String] = ["full", "tied", "diag", "spherical"]
    for t in range(len(cov_types)):
        var ctype = cov_types[t]
        var gmm = GaussianMixture[DType.float64](
            n_components=3, covariance_type=ctype, random_state=42
        )
        gmm.fit(X)

        var bytes = dumps(gmm)
        var restored = loads[GaussianMixture[DType.float64]](bytes)
        assert_true(restored.is_fitted)
        assert_equal(restored.n_features_in_, 2)
        assert_almost_equal(restored.lower_bound_, gmm.lower_bound_, atol=1e-12)

        var p1 = gmm.predict(X)
        var p2 = restored.predict(X)
        var pr1 = gmm.predict_proba(X)
        var pr2 = restored.predict_proba(X)
        for i in range(18):
            assert_equal(p1[i], p2[i])
            for k in range(3):
                assert_almost_equal(pr1[i, k], pr2[i, k], atol=1e-12)
        assert_almost_equal(gmm.score(X), restored.score(X), atol=1e-12)
        assert_almost_equal(gmm.aic(X), restored.aic(X), atol=1e-12)
        assert_almost_equal(gmm.bic(X), restored.bic(X), atol=1e-12)

    # Truncated buffer rejection
    var gmm_trunc = GaussianMixture[DType.float64](n_components=2)
    gmm_trunc.fit(X)
    var full_bytes = dumps(gmm_trunc)
    var half = List[UInt8]()
    for i in range(len(full_bytes) // 2):
        half.append(full_bytes[i])
    with assert_raises():
        _ = loads[GaussianMixture[DType.float64]](half)


def test_gmm_copy_semantics_and_float32() raises:
    var X1 = _make_two_clusters_2d()
    var gmm1 = GaussianMixture[DType.float64](
        n_components=2, covariance_type="full", random_state=42
    )
    gmm1.fit(X1)
    var orig_preds = gmm1.predict(X1)

    var gmm2 = gmm1.copy()
    assert_true(gmm2.is_fitted)
    assert_equal(gmm2.n_features_in_, 2)

    # Refit copy with 3D data
    var X2 = Matrix[DType.float64](8, 3, 0.0)
    for i in range(4):
        X2[i, 0] = Float64(i) * 0.1
        X2[i, 1] = Float64(i) * 0.1
        X2[i, 2] = Float64(i) * 0.1
        X2[4 + i, 0] = 8.0 + Float64(i) * 0.1
        X2[4 + i, 1] = 8.0 + Float64(i) * 0.1
        X2[4 + i, 2] = 8.0 + Float64(i) * 0.1
    gmm2.fit(X2)

    # Original gmm1 must remain untouched
    assert_equal(gmm1.n_features_in_, 2)
    assert_equal(gmm1.means_.cols, 2)
    var preds_after = gmm1.predict(X1)
    for i in range(12):
        assert_equal(preds_after[i], orig_preds[i])
    assert_equal(gmm2.n_features_in_, 3)

    # Float32 precision support
    var X_f32 = X1.cast[DType.float32]()
    var gmm_f32 = GaussianMixture[DType.float32](
        n_components=2, covariance_type="full", random_state=42
    )
    gmm_f32.fit(X_f32)
    assert_true(gmm_f32.is_fitted)
    var preds_f32 = gmm_f32.predict(X_f32)
    assert_true(preds_f32[0] != preds_f32[6])
    var probs_f32 = gmm_f32.predict_proba(X_f32)
    for i in range(12):
        var rsum = Float64(probs_f32[i, 0]) + Float64(probs_f32[i, 1])
        assert_almost_equal(rsum, 1.0, atol=1e-5)
    assert_false(isnan(Float64(gmm_f32.score(X_f32))))
    var samp_f32 = gmm_f32.sample(n_samples=20)
    assert_equal(samp_f32.X.rows, 20)
    assert_equal(samp_f32.X.cols, 2)


def test_gmm_error_handling_and_exception_safety() raises:
    # Invalid hyperparameters
    with assert_raises():
        _ = GaussianMixture[DType.float64](n_components=0)
    with assert_raises():
        _ = GaussianMixture[DType.float64](covariance_type="invalid")
    with assert_raises():
        _ = GaussianMixture[DType.float64](tol=-1e-3)
    with assert_raises():
        _ = GaussianMixture[DType.float64](tol=nan[DType.float64]())
    with assert_raises():
        _ = GaussianMixture[DType.float64](reg_covar=-1e-6)
    with assert_raises():
        _ = GaussianMixture[DType.float64](max_iter=0)
    with assert_raises():
        _ = GaussianMixture[DType.float64](n_init=0)
    with assert_raises():
        _ = GaussianMixture[DType.float64](init_params="invalid")

    var X = _make_two_clusters_2d()

    # Unfitted inference/scoring/sampling
    var unfitted = GaussianMixture[DType.float64](n_components=2)
    with assert_raises():
        _ = unfitted.predict(X)
    with assert_raises():
        _ = unfitted.predict_proba(X)
    with assert_raises():
        _ = unfitted.score_samples(X)
    with assert_raises():
        _ = unfitted.score(X)
    with assert_raises():
        _ = unfitted.aic(X)
    with assert_raises():
        _ = unfitted.bic(X)
    with assert_raises():
        _ = unfitted.sample(5)

    # N < n_components
    var X_one = Matrix[DType.float64](1, 2, 0.0)
    with assert_raises():
        unfitted.fit(X_one)

    # Invalid weights_init (wrong length, negative, non-unit sum)
    var bad_w_len: List[Scalar[DType.float64]] = [1.0]
    var gmm_w1 = GaussianMixture[DType.float64](
        n_components=2, weights_init=bad_w_len
    )
    with assert_raises():
        gmm_w1.fit(X)

    var bad_w_neg: List[Scalar[DType.float64]] = [-0.2, 1.2]
    var gmm_w2 = GaussianMixture[DType.float64](
        n_components=2, weights_init=bad_w_neg
    )
    with assert_raises():
        gmm_w2.fit(X)

    var bad_w_sum: List[Scalar[DType.float64]] = [0.2, 0.2]
    var gmm_w3 = GaussianMixture[DType.float64](
        n_components=2, weights_init=bad_w_sum
    )
    with assert_raises():
        gmm_w3.fit(X)

    # Invalid means_init shape
    var bad_means = Matrix[DType.float64](3, 2, 0.0)
    var gmm_m = GaussianMixture[DType.float64](
        n_components=2, means_init=bad_means
    )
    with assert_raises():
        gmm_m.fit(X)

    # Invalid precisions_init: asymmetric / non-SPD / wrong shape / non-positive diag
    var asym = Matrix[DType.float64](2, 2, 0.0)
    asym[0, 0] = 2.0
    asym[0, 1] = 1.0
    asym[1, 0] = 0.0
    asym[1, 1] = 2.0
    var p_asym = List[Matrix[DType.float64]]()
    p_asym.append(asym.copy())
    p_asym.append(asym.copy())
    var gmm_pa = GaussianMixture[DType.float64](
        n_components=2, covariance_type="full", precisions_init=p_asym
    )
    with assert_raises():
        gmm_pa.fit(X)

    var bad_p_diag = Matrix[DType.float64](2, 2, 1.0)
    bad_p_diag[1, 0] = 0.0
    var p_diag_list = List[Matrix[DType.float64]]()
    p_diag_list.append(bad_p_diag^)
    var gmm_pd = GaussianMixture[DType.float64](
        n_components=2, covariance_type="diag", precisions_init=p_diag_list
    )
    with assert_raises():
        gmm_pd.fit(X)

    # Fit valid model, then verify predict/sample errors & strong exception safety
    var gmm = GaussianMixture[DType.float64](
        n_components=2, covariance_type="full", random_state=42
    )
    gmm.fit(X)
    var saved_preds = gmm.predict(X)

    var X_wrong_dim = Matrix[DType.float64](4, 3, 0.0)
    with assert_raises():
        _ = gmm.predict(X_wrong_dim)
    with assert_raises():
        _ = gmm.sample(0)

    # Failing second fit must leave previously fitted model intact
    with assert_raises():
        gmm.fit(X_one)
    assert_true(gmm.is_fitted)
    assert_equal(gmm.n_features_in_, 2)
    var after_preds = gmm.predict(X)
    for i in range(12):
        assert_equal(after_preds[i], saved_preds[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
