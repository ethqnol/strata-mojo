from std.math import log, exp, isnan, isinf, nan, inf
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import (
    Matrix,
    Dataset,
    CategoricalNB,
    OrdinalEncoder,
    PipelineClassifier,
)
from strata.base import fit, predict, predict_proba
from strata.io.serializer import dumps, loads


def test_categorical_nb_binary_exact_math() raises:
    # 6 samples, 2 categorical features:
    # Feature 0 has categories {0, 1} -> n_categories = 2
    # Feature 1 has categories {0, 1, 2} -> n_categories = 3
    var X = Matrix[DType.float64](6, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[3, 0] = 1.0
    X[3, 1] = 2.0
    X[4, 0] = 1.0
    X[4, 1] = 2.0
    X[5, 0] = 0.0
    X[5, 1] = 2.0

    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var cnb = CategoricalNB[DType.float64](alpha=1.0)
    cnb.fit(X, y)

    assert_true(cnb.is_fitted)
    assert_equal(cnb.n_features_in_, 2)
    assert_equal(len(cnb.classes_), 2)
    assert_equal(cnb.classes_[0], 0)
    assert_equal(cnb.classes_[1], 1)
    assert_equal(cnb.class_count_[0], 3)
    assert_equal(cnb.class_count_[1], 3)
    assert_equal(len(cnb.n_categories_), 2)
    assert_equal(cnb.n_categories_[0], 2)
    assert_equal(cnb.n_categories_[1], 3)

    # Priors: log(0.5)
    assert_almost_equal(cnb.class_log_prior_[0], log(0.5), atol=1e-12)
    assert_almost_equal(cnb.class_log_prior_[1], log(0.5), atol=1e-12)

    # Feature 0 counts: class 0 -> [2, 1], class 1 -> [1, 2]
    assert_almost_equal(cnb.category_count_[0][0, 0], 2.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[0][0, 1], 1.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[0][1, 0], 1.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[0][1, 1], 2.0, atol=1e-12)

    # Feature 1 counts: class 0 -> [2, 1, 0], class 1 -> [0, 0, 3]
    assert_almost_equal(cnb.category_count_[1][0, 0], 2.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[1][0, 1], 1.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[1][0, 2], 0.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[1][1, 0], 0.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[1][1, 1], 0.0, atol=1e-12)
    assert_almost_equal(cnb.category_count_[1][1, 2], 3.0, atol=1e-12)

    # Smoothed feature log probabilities (alpha = 1.0)
    # Feat 0: denom = 3 + 1*2 = 5
    assert_almost_equal(
        cnb.feature_log_prob_[0][0, 0], log(3.0 / 5.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[0][0, 1], log(2.0 / 5.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[0][1, 0], log(2.0 / 5.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[0][1, 1], log(3.0 / 5.0), atol=1e-12
    )

    # Feat 1: denom = 3 + 1*3 = 6
    assert_almost_equal(
        cnb.feature_log_prob_[1][0, 0], log(3.0 / 6.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[1][0, 1], log(2.0 / 6.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[1][0, 2], log(1.0 / 6.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[1][1, 0], log(1.0 / 6.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[1][1, 1], log(1.0 / 6.0), atol=1e-12
    )
    assert_almost_equal(
        cnb.feature_log_prob_[1][1, 2], log(4.0 / 6.0), atol=1e-12
    )

    var X_test = Matrix[DType.float64](2, 2, 0.0)
    X_test[0, 0] = 0.0
    X_test[0, 1] = 0.0
    X_test[1, 0] = 1.0
    X_test[1, 1] = 2.0

    var preds = cnb.predict(X_test)
    assert_equal(len(preds), 2)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 1)

    var probs = cnb.predict_proba(X_test)
    assert_almost_equal(probs[0, 0], 9.0 / 11.0, atol=1e-12)
    assert_almost_equal(probs[0, 1], 2.0 / 11.0, atol=1e-12)
    assert_almost_equal(probs[1, 0], 1.0 / 7.0, atol=1e-12)
    assert_almost_equal(probs[1, 1], 6.0 / 7.0, atol=1e-12)

    var log_probs = cnb.predict_log_proba(X_test)
    assert_almost_equal(log_probs[0, 0], log(9.0 / 11.0), atol=1e-12)
    assert_almost_equal(log_probs[0, 1], log(2.0 / 11.0), atol=1e-12)
    assert_almost_equal(log_probs[1, 0], log(1.0 / 7.0), atol=1e-12)
    assert_almost_equal(log_probs[1, 1], log(6.0 / 7.0), atol=1e-12)


def test_categorical_nb_multiclass_noncontiguous_labels() raises:
    var X = Matrix[DType.float64](9, 2, 0.0)
    # Class 10
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 0.0
    X[2, 1] = 0.0
    # Class -1
    X[3, 0] = 1.0
    X[3, 1] = 2.0
    X[4, 0] = 1.0
    X[4, 1] = 2.0
    X[5, 0] = 1.0
    X[5, 1] = 1.0
    # Class 5
    X[6, 0] = 2.0
    X[6, 1] = 0.0
    X[7, 0] = 2.0
    X[7, 1] = 2.0
    X[8, 0] = 2.0
    X[8, 1] = 0.0

    var y: List[Scalar[DType.int32]] = [10, 10, 10, -1, -1, -1, 5, 5, 5]

    var cnb = CategoricalNB[DType.float64](alpha=1.0)
    cnb.fit(X, y)

    assert_equal(len(cnb.classes_), 3)
    assert_equal(cnb.classes_[0], -1)
    assert_equal(cnb.classes_[1], 5)
    assert_equal(cnb.classes_[2], 10)

    var preds = cnb.predict(X)
    for i in range(9):
        assert_equal(preds[i], Int(y[i]))

    var probs = cnb.predict_proba(X)
    assert_equal(probs.rows, 9)
    assert_equal(probs.cols, 3)
    for i in range(9):
        var row_sum = probs[i, 0] + probs[i, 1] + probs[i, 2]
        assert_almost_equal(row_sum, 1.0, atol=1e-12)


def test_categorical_nb_alpha_smoothing_and_zero_alpha() raises:
    var X = Matrix[DType.float64](6, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[3, 0] = 1.0
    X[3, 1] = 2.0
    X[4, 0] = 1.0
    X[4, 1] = 2.0
    X[5, 0] = 0.0
    X[5, 1] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    # Partial smoothing alpha = 0.5
    var cnb_half = CategoricalNB[DType.float64](alpha=0.5)
    cnb_half.fit(X, y)
    # Feat 1 for class 0: counts [2, 1, 0], denom = 3 + 0.5*3 = 4.5
    assert_almost_equal(
        cnb_half.feature_log_prob_[1][0, 0], log(2.5 / 4.5), atol=1e-12
    )
    assert_almost_equal(
        cnb_half.feature_log_prob_[1][0, 2], log(0.5 / 4.5), atol=1e-12
    )

    # Zero smoothing alpha = 0.0 with zero-count category in one class
    var cnb_zero = CategoricalNB[DType.float64](
        alpha=0.0, min_categories=[2, 4]
    )
    cnb_zero.fit(X, y)
    assert_almost_equal(
        cnb_zero.feature_log_prob_[1][0, 0], log(2.0 / 3.0), atol=1e-12
    )
    assert_almost_equal(cnb_zero.feature_log_prob_[1][1, 2], 0.0, atol=1e-12)

    var X_probe = Matrix[DType.float64](2, 2, 0.0)
    X_probe[0, 0] = 0.0
    X_probe[0, 1] = 0.0  # Seen in class 0, zero count in class 1
    X_probe[1, 0] = 0.0
    X_probe[1, 1] = 3.0  # Zero count in BOTH classes via min_categories

    var preds = cnb_zero.predict(X_probe)
    assert_equal(preds[0], 0)

    var probs = cnb_zero.predict_proba(X_probe)
    assert_almost_equal(probs[0, 0], 1.0, atol=1e-12)
    assert_almost_equal(probs[0, 1], 0.0, atol=1e-12)
    for i in range(2):
        for k in range(2):
            assert_false(isnan(probs[i, k]))
            assert_false(isinf(probs[i, k]))
        assert_almost_equal(probs[i, 0] + probs[i, 1], 1.0, atol=1e-12)


def test_categorical_nb_min_categories() raises:
    # Training only observes {0, 1} for both features
    var X_train = Matrix[DType.float64](4, 2, 0.0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 0.0
    X_train[1, 1] = 1.0
    X_train[2, 0] = 1.0
    X_train[2, 1] = 1.0
    X_train[3, 0] = 1.0
    X_train[3, 1] = 0.0
    var y_train: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    # Uniform min_categories = [5]
    var cnb_uniform = CategoricalNB[DType.float64](
        alpha=1.0, min_categories=[5]
    )
    cnb_uniform.fit(X_train, y_train)
    assert_equal(cnb_uniform.n_categories_[0], 5)
    assert_equal(cnb_uniform.n_categories_[1], 5)
    assert_equal(cnb_uniform.category_count_[0].cols, 5)
    assert_equal(cnb_uniform.feature_log_prob_[1].cols, 5)

    # Test predicting on unseen category 3 (within [0, 5))
    var X_unseen = Matrix[DType.float64](1, 2, 0.0)
    X_unseen[0, 0] = 0.0
    X_unseen[0, 1] = 3.0
    var preds = cnb_uniform.predict(X_unseen)
    assert_equal(preds[0], 0)

    # Per-feature min_categories = [4, 6]
    var cnb_per_feat = CategoricalNB[DType.float64](
        alpha=1.0, min_categories=[4, 6]
    )
    cnb_per_feat.fit(X_train, y_train)
    assert_equal(cnb_per_feat.n_categories_[0], 4)
    assert_equal(cnb_per_feat.n_categories_[1], 6)
    # Feat 0 denom for class 0: 2 + 1*4 = 6; cat 3 has count 0 -> log(1/6)
    assert_almost_equal(
        cnb_per_feat.feature_log_prob_[0][0, 3], log(1.0 / 6.0), atol=1e-12
    )
    # Feat 1 denom for class 1: 2 + 1*6 = 8; cat 5 has count 0 -> log(1/8)
    assert_almost_equal(
        cnb_per_feat.feature_log_prob_[1][1, 5], log(1.0 / 8.0), atol=1e-12
    )


def test_categorical_nb_priors_and_fit_prior_false() raises:
    # Imbalanced dataset: 4 samples of class 0, 1 sample of class 1
    var X = Matrix[DType.float64](5, 1, 0.0)
    X[0, 0] = 0.0
    X[1, 0] = 0.0
    X[2, 0] = 0.0
    X[3, 0] = 1.0
    X[4, 0] = 1.0
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 0, 1]

    # fit_prior = False -> uniform prior log(0.5)
    var cnb_unif = CategoricalNB[DType.float64](alpha=1.0, fit_prior=False)
    cnb_unif.fit(X, y)
    assert_almost_equal(cnb_unif.class_log_prior_[0], -log(2.0), atol=1e-12)
    assert_almost_equal(cnb_unif.class_log_prior_[1], -log(2.0), atol=1e-12)

    # Explicit class_prior = [0.8, 0.2]
    var priors: List[Scalar[DType.float64]] = [0.8, 0.2]
    var cnb_custom = CategoricalNB[DType.float64](alpha=1.0, class_prior=priors)
    cnb_custom.fit(X, y)
    assert_almost_equal(cnb_custom.class_log_prior_[0], log(0.8), atol=1e-12)
    assert_almost_equal(cnb_custom.class_log_prior_[1], log(0.2), atol=1e-12)


def test_categorical_nb_constant_and_gap_categories() raises:
    # Feature 0 is constant (all 0), Feature 1 has sparse categories {0, 3}
    var X = Matrix[DType.float64](4, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 3.0
    X[3, 0] = 0.0
    X[3, 1] = 3.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var cnb = CategoricalNB[DType.float64](alpha=1.0)
    cnb.fit(X, y)

    assert_equal(cnb.n_categories_[0], 1)
    assert_equal(cnb.n_categories_[1], 4)
    # Constant feature must have log prob == 0.0
    assert_almost_equal(cnb.feature_log_prob_[0][0, 0], 0.0, atol=1e-12)
    assert_almost_equal(cnb.feature_log_prob_[0][1, 0], 0.0, atol=1e-12)

    # Unobserved gap category 2 in feature 1 is still valid in [0, 4)
    var X_gap = Matrix[DType.float64](1, 2, 0.0)
    X_gap[0, 0] = 0.0
    X_gap[0, 1] = 2.0
    var probs = cnb.predict_proba(X_gap)
    assert_almost_equal(probs[0, 0], 0.5, atol=1e-12)
    assert_almost_equal(probs[0, 1], 0.5, atol=1e-12)


def test_categorical_nb_integer_matrix_and_float32() raises:
    # Int32 matrix
    var X_i32 = Matrix[DType.int32](4, 2, 0)
    X_i32[0, 0] = 0
    X_i32[0, 1] = 1
    X_i32[1, 0] = 0
    X_i32[1, 1] = 0
    X_i32[2, 0] = 2
    X_i32[2, 1] = 1
    X_i32[3, 0] = 2
    X_i32[3, 1] = 2
    var y: List[Scalar[DType.int64]] = [0, 0, 1, 1]

    var cnb = CategoricalNB[DType.float64]()
    cnb.fit(X_i32, y)
    var preds_i32 = cnb.predict(X_i32)
    for i in range(4):
        assert_equal(preds_i32[i], Int(y[i]))

    # predict_proba / predict_log_proba on integer Matrix must raise DataConversionError
    with assert_raises():
        _ = cnb.predict_proba(X_i32)
    with assert_raises():
        _ = cnb.predict_log_proba(X_i32)

    # Int64 matrix
    var X_i64 = Matrix[DType.int64](4, 2, 0)
    X_i64[0, 0] = 0
    X_i64[0, 1] = 1
    X_i64[1, 0] = 0
    X_i64[1, 1] = 0
    X_i64[2, 0] = 2
    X_i64[2, 1] = 1
    X_i64[3, 0] = 2
    X_i64[3, 1] = 2
    var cnb_i64 = CategoricalNB[DType.float64]()
    cnb_i64.fit(X_i64, y)
    var preds_i64 = cnb_i64.predict(X_i64)
    for i in range(4):
        assert_equal(preds_i64[i], Int(y[i]))

    # Small integer dtypes (uint8 / int8)
    var X_u8 = Matrix[DType.uint8](4, 2, 0)
    X_u8[0, 0] = 0
    X_u8[0, 1] = 1
    X_u8[1, 0] = 0
    X_u8[1, 1] = 0
    X_u8[2, 0] = 5
    X_u8[2, 1] = 1
    X_u8[3, 0] = 5
    X_u8[3, 1] = 2
    var cnb_u8 = CategoricalNB[DType.float64]()
    cnb_u8.fit(X_u8, y)
    var preds_u8 = cnb_u8.predict(X_u8)
    for i in range(4):
        assert_equal(preds_u8[i], Int(y[i]))

    # Float32 compute_dtype with Float32 Matrix
    var X_f32 = Matrix[DType.float32](4, 2, 0.0)
    X_f32[0, 0] = 0.0
    X_f32[0, 1] = 0.0
    X_f32[1, 0] = 0.0
    X_f32[1, 1] = 1.0
    X_f32[2, 0] = 2.0
    X_f32[2, 1] = 1.0
    X_f32[3, 0] = 2.0
    X_f32[3, 1] = 2.0
    var cnb_f32 = CategoricalNB[DType.float32](alpha=1.0)
    cnb_f32.fit(X_f32, y)
    var preds_f32 = cnb_f32.predict(X_f32)
    for i in range(4):
        assert_equal(preds_f32[i], Int(y[i]))

    var probs_f32 = cnb_f32.predict_proba(X_f32)
    for i in range(4):
        var rsum = Float64(probs_f32[i, 0]) + Float64(probs_f32[i, 1])
        assert_almost_equal(rsum, 1.0, atol=1e-5)


def test_categorical_nb_pipeline_and_dataset_integration() raises:
    # Arbitrary numeric category codes mapped to 0..K-1 via OrdinalEncoder
    var X = Matrix[DType.float64](6, 2, 0.0)
    X[0, 0] = 10.0
    X[0, 1] = 100.0
    X[1, 0] = 10.0
    X[1, 1] = 200.0
    X[2, 0] = 10.0
    X[2, 1] = 100.0
    X[3, 0] = 30.0
    X[3, 1] = 300.0
    X[4, 0] = 30.0
    X[4, 1] = 300.0
    X[5, 0] = 20.0
    X[5, 1] = 300.0

    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var pipe = PipelineClassifier(
        OrdinalEncoder[DType.float64](), CategoricalNB[DType.float64]()
    )
    pipe.fit(X, y)
    var pipe_preds = pipe.predict(X)
    var pipe_probs = pipe.predict_proba(X)
    for i in range(6):
        assert_equal(pipe_preds[i], Int(y[i]))
        assert_almost_equal(
            pipe_probs[i, 0] + pipe_probs[i, 1], 1.0, atol=1e-12
        )

    # Dataset integration using free functions from strata.base
    var X_ds = Matrix[DType.float64](4, 2, 0.0)
    X_ds[0, 0] = 0.0
    X_ds[0, 1] = 0.0
    X_ds[1, 0] = 0.0
    X_ds[1, 1] = 1.0
    X_ds[2, 0] = 1.0
    X_ds[2, 1] = 2.0
    X_ds[3, 0] = 1.0
    X_ds[3, 1] = 2.0
    var y_ds: List[Scalar[DType.int32]] = [0, 0, 1, 1]
    var ds = Dataset[DType.float64, DType.int32](X_ds^, y_ds^)

    var clf = CategoricalNB[DType.float64](alpha=1.0)
    fit(clf, ds)
    var ds_preds = predict(clf, ds)
    var ds_probs = predict_proba(clf, ds)
    assert_equal(ds_preds[0], 0)
    assert_equal(ds_preds[1], 0)
    assert_equal(ds_preds[2], 1)
    assert_equal(ds_preds[3], 1)
    assert_equal(ds_probs.rows, 4)
    assert_equal(ds_probs.cols, 2)


def test_categorical_nb_serialization() raises:
    # Unfitted round-trip
    var unfitted = CategoricalNB[DType.float64](
        alpha=0.25, fit_prior=False, min_categories=[3, 5]
    )
    var unfitted_bytes = dumps(unfitted)
    var unfitted_restored = loads[CategoricalNB[DType.float64]](unfitted_bytes)
    assert_false(unfitted_restored.is_fitted)
    assert_almost_equal(unfitted_restored.alpha, 0.25, atol=1e-12)
    assert_false(unfitted_restored.fit_prior)
    assert_equal(len(unfitted_restored.min_categories), 2)
    assert_equal(unfitted_restored.min_categories[0], 3)
    assert_equal(unfitted_restored.min_categories[1], 5)

    # Fitted round-trip
    var X = Matrix[DType.float64](6, 2, 0.0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[3, 0] = 1.0
    X[3, 1] = 2.0
    X[4, 0] = 2.0
    X[4, 1] = 2.0
    X[5, 0] = 2.0
    X[5, 1] = 1.0
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var cnb = CategoricalNB[DType.float64](alpha=0.5, min_categories=[4, 4])
    cnb.fit(X, y)

    var bytes = dumps(cnb)
    var restored = loads[CategoricalNB[DType.float64]](bytes)

    assert_true(restored.is_fitted)
    assert_equal(restored.n_features_in_, cnb.n_features_in_)
    assert_almost_equal(restored.alpha, cnb.alpha, atol=1e-12)
    assert_equal(len(restored.n_categories_), 2)
    assert_equal(restored.n_categories_[0], 4)
    assert_equal(restored.n_categories_[1], 4)

    var p1 = cnb.predict(X)
    var p2 = restored.predict(X)
    var pr1 = cnb.predict_proba(X)
    var pr2 = restored.predict_proba(X)
    for i in range(6):
        assert_equal(p1[i], p2[i])
        assert_almost_equal(pr1[i, 0], pr2[i, 0], atol=1e-12)
        assert_almost_equal(pr1[i, 1], pr2[i, 1], atol=1e-12)

    # Truncated buffer rejection
    var truncated = List[UInt8]()
    for i in range(len(bytes) // 2):
        truncated.append(bytes[i])
    with assert_raises():
        _ = loads[CategoricalNB[DType.float64]](truncated)


def test_categorical_nb_copy_semantics_and_refit() raises:
    var X1 = Matrix[DType.float64](4, 2, 0.0)
    X1[0, 0] = 0.0
    X1[0, 1] = 0.0
    X1[1, 0] = 0.0
    X1[1, 1] = 1.0
    X1[2, 0] = 1.0
    X1[2, 1] = 0.0
    X1[3, 0] = 1.0
    X1[3, 1] = 1.0
    var y1: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var cnb1 = CategoricalNB[DType.float64](alpha=1.0)
    cnb1.fit(X1, y1)

    var cnb2 = cnb1.copy()
    assert_true(cnb2.is_fitted)
    assert_equal(cnb2.n_features_in_, 2)

    # Refit cnb2 with different feature dimensionality and flipped labels
    var X2 = Matrix[DType.float64](4, 3, 0.0)
    X2[0, 0] = 0.0
    X2[0, 1] = 0.0
    X2[0, 2] = 0.0
    X2[1, 0] = 0.0
    X2[1, 1] = 1.0
    X2[1, 2] = 0.0
    X2[2, 0] = 1.0
    X2[2, 1] = 0.0
    X2[2, 2] = 2.0
    X2[3, 0] = 1.0
    X2[3, 1] = 1.0
    X2[3, 2] = 2.0
    var y2: List[Scalar[DType.int32]] = [1, 1, 0, 0]
    cnb2.fit(X2, y2)

    # cnb1 must remain completely isolated
    assert_equal(cnb1.n_features_in_, 2)
    assert_equal(len(cnb1.category_count_), 2)
    assert_equal(cnb1.predict(X1)[0], 0)
    assert_equal(cnb1.predict(X1)[2], 1)

    assert_equal(cnb2.n_features_in_, 3)
    assert_equal(len(cnb2.category_count_), 3)
    assert_equal(cnb2.predict(X2)[0], 1)
    assert_equal(cnb2.predict(X2)[2], 0)


def test_categorical_nb_error_handling_and_exception_safety() raises:
    # Invalid alpha
    with assert_raises():
        _ = CategoricalNB[DType.float64](alpha=-0.1)
    with assert_raises():
        _ = CategoricalNB[DType.float64](alpha=nan[DType.float64]())
    with assert_raises():
        _ = CategoricalNB[DType.float64](alpha=inf[DType.float64]())

    # Invalid min_categories (< 1)
    with assert_raises():
        _ = CategoricalNB[DType.float64](min_categories=[0])

    # Unfitted predict / predict_proba / predict_log_proba
    var unfitted = CategoricalNB[DType.float64]()
    var X_ok = Matrix[DType.float64](4, 2, 0.0)
    X_ok[0, 0] = 0.0
    X_ok[0, 1] = 0.0
    X_ok[1, 0] = 0.0
    X_ok[1, 1] = 1.0
    X_ok[2, 0] = 1.0
    X_ok[2, 1] = 0.0
    X_ok[3, 0] = 1.0
    X_ok[3, 1] = 1.0
    var y_ok: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    with assert_raises():
        _ = unfitted.predict(X_ok)
    with assert_raises():
        _ = unfitted.predict_proba(X_ok)
    with assert_raises():
        _ = unfitted.predict_log_proba(X_ok)

    # Wrong min_categories length during fit
    var cnb_bad_min_len = CategoricalNB[DType.float64](min_categories=[2, 2, 2])
    with assert_raises():
        cnb_bad_min_len.fit(X_ok, y_ok)

    # Negative values in fit
    var X_neg = X_ok.copy()
    X_neg[1, 0] = -1.0
    var cnb = CategoricalNB[DType.float64]()
    with assert_raises():
        cnb.fit(X_neg, y_ok)

    # Fractional float values in fit
    var X_frac = X_ok.copy()
    X_frac[2, 1] = 1.5
    with assert_raises():
        cnb.fit(X_frac, y_ok)

    # Single class rejection
    var y_single: List[Scalar[DType.int32]] = [1, 1, 1, 1]
    with assert_raises():
        cnb.fit(X_ok, y_single)

    # Invalid class_prior (wrong length, non-positive, non-unit sum)
    var bad_priors_len: List[Scalar[DType.float64]] = [0.5, 0.25, 0.25]
    var cnb_p1 = CategoricalNB[DType.float64](class_prior=bad_priors_len)
    with assert_raises():
        cnb_p1.fit(X_ok, y_ok)

    var bad_priors_zero: List[Scalar[DType.float64]] = [1.0, 0.0]
    var cnb_p2 = CategoricalNB[DType.float64](class_prior=bad_priors_zero)
    with assert_raises():
        cnb_p2.fit(X_ok, y_ok)

    var bad_priors_sum: List[Scalar[DType.float64]] = [0.3, 0.3]
    var cnb_p3 = CategoricalNB[DType.float64](class_prior=bad_priors_sum)
    with assert_raises():
        cnb_p3.fit(X_ok, y_ok)

    # Fit valid model, then verify predict-time rejections & strong exception safety
    cnb.fit(X_ok, y_ok)

    # Negative value at predict
    with assert_raises():
        _ = cnb.predict(X_neg)

    # Fractional value at predict
    with assert_raises():
        _ = cnb.predict(X_frac)

    # Unseen out-of-range category at predict
    var X_oob = Matrix[DType.float64](1, 2, 0.0)
    X_oob[0, 0] = 0.0
    X_oob[0, 1] = 2.0
    with assert_raises():
        _ = cnb.predict(X_oob)

    # Dimension mismatch at predict
    var X_dim = Matrix[DType.float64](2, 3, 0.0)
    with assert_raises():
        _ = cnb.predict(X_dim)

    # Strong exception safety: failing second fit must preserve first fitted state
    with assert_raises():
        cnb.fit(X_frac, y_ok)
    assert_true(cnb.is_fitted)
    assert_equal(cnb.n_features_in_, 2)
    var preserved_preds = cnb.predict(X_ok)
    for i in range(4):
        assert_equal(preserved_preds[i], Int(y_ok[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
