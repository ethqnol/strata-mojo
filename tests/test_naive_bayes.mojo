from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import (
    Matrix,
    CSRMatrix,
    Dataset,
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
    StandardScaler,
    PipelineClassifier,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
    BufferWriter,
    BufferReader,
)
from strata.base import fit, predict, predict_proba
from strata.io.serializer import dumps, loads


def test_gaussian_nb_binary() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -2.5
    X[1, 1] = -1.5
    X[2, 0] = -1.5
    X[2, 1] = -2.0
    X[3, 0] = 2.0
    X[3, 1] = 2.0
    X[4, 0] = 1.5
    X[4, 1] = 2.5
    X[5, 0] = 2.5
    X[5, 1] = 1.5

    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var gnb = GaussianNB[DType.float64]()
    gnb.fit(X, y)

    assert_true(gnb.is_fitted)
    assert_equal(len(gnb.classes_), 2)
    assert_equal(gnb.classes_[0], 0)
    assert_equal(gnb.classes_[1], 1)
    assert_equal(gnb.class_count_[0], 3)
    assert_equal(gnb.class_count_[1], 3)

    var preds = gnb.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))

    var probs = gnb.predict_proba(X)
    assert_equal(probs.rows, 6)
    assert_equal(probs.cols, 2)
    for i in range(6):
        var row_sum = probs[i, 0] + probs[i, 1]
        assert_almost_equal(row_sum, 1.0, rtol=1e-4)
        assert_true(probs[i, Int(y[i])] > 0.8)


def test_gaussian_nb_multiclass() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -5.0
    X[0, 1] = -5.0  # class 0
    X[1, 0] = -4.5
    X[1, 1] = -5.5  # class 0
    X[2, 0] = 0.0
    X[2, 1] = 0.0  # class 1
    X[3, 0] = 0.5
    X[3, 1] = -0.5  # class 1
    X[4, 0] = 5.0
    X[4, 1] = 5.0  # class 2
    X[5, 0] = 5.5
    X[5, 1] = 4.5  # class 2

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1, 2, 2]

    var gnb = GaussianNB[DType.float64]()
    gnb.fit(X, y)

    assert_equal(len(gnb.classes_), 3)
    assert_equal(gnb.classes_[0], 0)
    assert_equal(gnb.classes_[1], 1)
    assert_equal(gnb.classes_[2], 2)

    var preds = gnb.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))

    var probs = gnb.predict_proba(X)
    for i in range(6):
        var row_sum = probs[i, 0] + probs[i, 1] + probs[i, 2]
        assert_almost_equal(row_sum, 1.0, rtol=1e-4)


def test_gaussian_nb_var_smoothing_constant_feature() raises:
    # Feature 1 has 0 variance (constant value 1.0 across all samples)
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = 1.0
    X[1, 0] = -1.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[3, 0] = 2.0
    X[3, 1] = 1.0

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var gnb = GaussianNB[DType.float64](var_smoothing=1e-5)
    gnb.fit(X, y)

    # Variance on constant feature must be positive due to smoothing epsilon
    assert_true(gnb.var_[0, 1] > 0)
    assert_true(gnb.var_[1, 1] > 0)

    var preds = gnb.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_gaussian_nb_custom_priors() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -1.0
    X[1, 0] = -0.5
    X[2, 0] = 0.5
    X[3, 0] = 1.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var priors: List[Scalar[DType.float64]] = [0.99, 0.01]
    var gnb = GaussianNB[DType.float64](priors=priors)
    gnb.fit(X, y)

    assert_almost_equal(gnb.class_prior_[0], 0.99, rtol=1e-4)
    assert_almost_equal(gnb.class_prior_[1], 0.01, rtol=1e-4)

    # Point near decision boundary (e.g. 0.0) should be pushed heavily to class 0 by prior
    var test_pt = Matrix[DType.float64](1, 1, 0)
    test_pt[0, 0] = 0.0
    var prob = gnb.predict_proba(test_pt)
    assert_true(prob[0, 0] > prob[0, 1])


def test_gaussian_nb_pipeline() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 10.0
    X[0, 1] = 100.0
    X[1, 0] = 12.0
    X[1, 1] = 110.0
    X[2, 0] = 11.0
    X[2, 1] = 105.0
    X[3, 0] = 50.0
    X[3, 1] = 500.0
    X[4, 0] = 52.0
    X[4, 1] = 510.0
    X[5, 0] = 51.0
    X[5, 1] = 505.0
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    var scaler = StandardScaler[DType.float64]()
    var gnb = GaussianNB[DType.float64]()
    var pipe = PipelineClassifier(scaler.copy(), gnb.copy())
    pipe.fit(X, y)

    var preds = pipe.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))


def test_gaussian_nb_errors() raises:
    var gnb = GaussianNB[DType.float64]()
    var X = Matrix[DType.float64](2, 2, 0)

    # Not fitted error
    with assert_raises():
        _ = gnb.predict(X)

    # Single class error
    var y_single: List[Scalar[DType.int32]] = [0, 0]
    with assert_raises():
        gnb.fit(X, y_single)

    # Negative var_smoothing
    with assert_raises():
        _ = GaussianNB[DType.float64](var_smoothing=-1.0)


def test_gaussian_nb_serialization() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -1.0
    X[1, 0] = -1.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 1.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var gnb = GaussianNB[DType.float64](var_smoothing=1e-8)
    gnb.fit(X, y)
    var orig_preds = gnb.predict(X)

    var writer = BufferWriter()
    gnb.serialize(writer)

    var reader = BufferReader(writer.get_bytes())
    var loaded = GaussianNB[DType.float64].deserialize(reader)

    assert_true(loaded.is_fitted)
    assert_equal(len(loaded.classes_), len(gnb.classes_))
    assert_almost_equal(loaded.theta_[0, 0], gnb.theta_[0, 0], rtol=1e-5)
    assert_almost_equal(loaded.var_[0, 0], gnb.var_[0, 0], rtol=1e-5)

    var loaded_preds = loaded.predict(X)
    for i in range(4):
        assert_equal(orig_preds[i], loaded_preds[i])


def test_multinomial_nb_dense() raises:
    # 4 documents with 3 vocabulary words [word_tech, word_sports, word_common]
    var X = Matrix[DType.float64](4, 3, 0)
    X[0, 0] = 3.0
    X[0, 1] = 0.0
    X[0, 2] = 1.0  # Doc 0: Tech
    X[1, 0] = 2.0
    X[1, 1] = 0.0
    X[1, 2] = 2.0  # Doc 1: Tech
    X[2, 0] = 0.0
    X[2, 1] = 4.0
    X[2, 2] = 1.0  # Doc 2: Sports
    X[3, 0] = 0.0
    X[3, 1] = 3.0
    X[3, 2] = 2.0  # Doc 3: Sports

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var mnb = MultinomialNB[DType.float64](alpha=1.0)
    mnb.fit(X, y)

    assert_true(mnb.is_fitted)
    assert_equal(len(mnb.classes_), 2)
    assert_equal(mnb.classes_[0], 0)
    assert_equal(mnb.classes_[1], 1)

    var preds = mnb.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)

    # Test on unseen tech document [4, 0, 1]
    var test_tech = Matrix[DType.float64](1, 3, 0)
    test_tech[0, 0] = 4.0
    test_tech[0, 1] = 0.0
    test_tech[0, 2] = 1.0
    assert_equal(mnb.predict(test_tech)[0], 0)

    # Test on unseen sports document [0, 5, 1]
    var test_sports = Matrix[DType.float64](1, 3, 0)
    test_sports[0, 0] = 0.0
    test_sports[0, 1] = 5.0
    test_sports[0, 2] = 1.0
    assert_equal(mnb.predict(test_sports)[0], 1)

    var probs = mnb.predict_proba(test_tech)
    assert_almost_equal(probs[0, 0] + probs[0, 1], 1.0, rtol=1e-4)
    assert_true(probs[0, 0] > 0.8)


def test_multinomial_nb_sparse_parity() raises:
    # Dense matrix
    var X_dense = Matrix[DType.float64](4, 3, 0)
    X_dense[0, 0] = 3.0
    X_dense[0, 1] = 0.0
    X_dense[0, 2] = 1.0
    X_dense[1, 0] = 2.0
    X_dense[1, 1] = 0.0
    X_dense[1, 2] = 2.0
    X_dense[2, 0] = 0.0
    X_dense[2, 1] = 4.0
    X_dense[2, 2] = 1.0
    X_dense[3, 0] = 0.0
    X_dense[3, 1] = 3.0
    X_dense[3, 2] = 2.0

    # Equivalent CSRMatrix
    var data: List[Scalar[DType.float64]] = [
        3.0,
        1.0,
        2.0,
        2.0,
        4.0,
        1.0,
        3.0,
        2.0,
    ]
    var indices: List[Int] = [0, 2, 0, 2, 1, 2, 1, 2]
    var indptr: List[Int] = [0, 2, 4, 6, 8]
    var X_sparse = CSRMatrix[DType.float64](4, 3, data^, indices^, indptr^)

    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var mnb_dense = MultinomialNB[DType.float64](alpha=1.0)
    mnb_dense.fit(X_dense, y)

    var mnb_sparse = MultinomialNB[DType.float64](alpha=1.0)
    mnb_sparse.fit(X_sparse, y)

    # Feature counts must match
    for k in range(2):
        for j in range(3):
            assert_almost_equal(
                mnb_dense.feature_count_[k, j],
                mnb_sparse.feature_count_[k, j],
                rtol=1e-5,
            )
            assert_almost_equal(
                mnb_dense.feature_log_prob_[k, j],
                mnb_sparse.feature_log_prob_[k, j],
                rtol=1e-5,
            )

    var preds_dense = mnb_dense.predict(X_dense)
    var preds_sparse = mnb_sparse.predict(X_sparse)
    for i in range(4):
        assert_equal(preds_dense[i], preds_sparse[i])


def test_multinomial_nb_smoothing_values() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 3.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1]

    # Test Lidstone smoothing (alpha = 0.5)
    var mnb_lidstone = MultinomialNB[DType.float64](alpha=0.5)
    mnb_lidstone.fit(X, y)
    assert_true(mnb_lidstone.is_fitted)

    # Test no smoothing (alpha = 0.0)
    var mnb_zero = MultinomialNB[DType.float64](alpha=0.0)
    mnb_zero.fit(X, y)
    assert_true(mnb_zero.is_fitted)


def test_multinomial_nb_negative_values_error() raises:
    var X = Matrix[DType.float64](2, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = -0.5  # Negative value
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    var y: List[Scalar[DType.int32]] = [0, 1]

    var mnb = MultinomialNB[DType.float64]()
    with assert_raises():
        mnb.fit(X, y)


def test_multinomial_nb_serialization() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 2.0
    X[0, 1] = 1.0
    X[1, 0] = 3.0
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 4.0
    X[3, 0] = 1.0
    X[3, 1] = 5.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var mnb = MultinomialNB[DType.float64](alpha=0.8)
    mnb.fit(X, y)
    var orig_preds = mnb.predict(X)

    var writer = BufferWriter()
    mnb.serialize(writer)

    var reader = BufferReader(writer.get_bytes())
    var loaded = MultinomialNB[DType.float64].deserialize(reader)

    assert_true(loaded.is_fitted)
    assert_almost_equal(loaded.alpha, 0.8, rtol=1e-5)
    assert_equal(len(loaded.classes_), len(mnb.classes_))

    var loaded_preds = loaded.predict(X)
    for i in range(4):
        assert_equal(orig_preds[i], loaded_preds[i])


def test_dataset_interface() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -1.0
    X[1, 1] = -1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[3, 0] = 2.0
    X[3, 1] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    var ds = Dataset[DType.float64, DType.int32](X.copy(), y.copy())

    var gnb = GaussianNB[DType.float64]()
    fit(gnb, ds)
    var preds = predict(gnb, ds)
    for i in range(4):
        assert_equal(preds[i], Int(y[i]))


def test_naive_bayes_arbitrary_signed_labels() raises:
    # Non-consecutive and negative class labels: -10 and +42
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -1.0
    X[1, 0] = -3.0
    X[1, 1] = -2.0
    X[2, 0] = 2.0
    X[2, 1] = 3.0
    X[3, 0] = 3.0
    X[3, 1] = 2.0
    var y: List[Scalar[DType.int32]] = [-10, -10, 42, 42]

    var gnb = GaussianNB[DType.float64]()
    gnb.fit(X, y)
    assert_equal(len(gnb.classes_), 2)
    assert_equal(gnb.classes_[0], -10)
    assert_equal(gnb.classes_[1], 42)

    var preds = gnb.predict(X)
    assert_equal(preds[0], -10)
    assert_equal(preds[1], -10)
    assert_equal(preds[2], 42)
    assert_equal(preds[3], 42)


def test_multinomial_nb_invalid_priors_and_dimension_mismatch() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 0.0
    X[2, 1] = 3.0
    X[3, 0] = 1.0
    X[3, 1] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]

    # Invalid priors: length mismatch
    var bad_priors_len: List[Scalar[DType.float64]] = [0.5, 0.3, 0.2]
    var mnb_bad_len = MultinomialNB[DType.float64](class_prior=bad_priors_len)
    with assert_raises():
        mnb_bad_len.fit(X, y)

    # Invalid priors: sum != 1.0
    var bad_priors_sum: List[Scalar[DType.float64]] = [0.8, 0.8]
    var mnb_bad_sum = MultinomialNB[DType.float64](class_prior=bad_priors_sum)
    with assert_raises():
        mnb_bad_sum.fit(X, y)

    # Valid model dimension mismatch at predict time
    var mnb = MultinomialNB[DType.float64]()
    mnb.fit(X, y)
    var X_wrong_dim = Matrix[DType.float64](2, 3, 0)
    with assert_raises():
        _ = mnb.predict(X_wrong_dim)


def test_bernoulli_nb_binary_dense() raises:
    # 4 samples, 4 binary features
    var X = Matrix[DType.float64](4, 4, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[0, 2] = 1.0
    X[0, 3] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[1, 2] = 0.0
    X[1, 3] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[2, 2] = 1.0
    X[2, 3] = 0.0
    X[3, 0] = 0.0
    X[3, 1] = 0.0
    X[3, 2] = 1.0
    X[3, 3] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var bnb = BernoulliNB[DType.float64](alpha=1.0, binarize=0.0)
    bnb.fit(X, y)

    assert_true(bnb.is_fitted)
    assert_equal(len(bnb.classes_), 2)
    assert_equal(bnb.classes_[0], 0)
    assert_equal(bnb.classes_[1], 1)

    var preds = bnb.predict(X)
    for i in range(4):
        assert_equal(preds[i], Int(y[i]))

    var probs = bnb.predict_proba(X)
    assert_equal(probs.rows, 4)
    assert_equal(probs.cols, 2)
    for i in range(4):
        var row_sum = probs[i, 0] + probs[i, 1]
        assert_almost_equal(row_sum, 1.0, rtol=1e-4)


def test_bernoulli_nb_multiclass() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    # Class 0: feature 0 active
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[0, 2] = 0.0
    X[1, 0] = 1.0
    X[1, 1] = 0.0
    X[1, 2] = 0.0
    # Class 1: feature 1 active
    X[2, 0] = 0.0
    X[2, 1] = 1.0
    X[2, 2] = 0.0
    X[3, 0] = 0.0
    X[3, 1] = 1.0
    X[3, 2] = 0.0
    # Class 2: feature 2 active
    X[4, 0] = 0.0
    X[4, 1] = 0.0
    X[4, 2] = 1.0
    X[5, 0] = 0.0
    X[5, 1] = 0.0
    X[5, 2] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)
    y.append(2)
    y.append(2)

    var bnb = BernoulliNB[DType.float64](alpha=1.0)
    bnb.fit(X, y)

    assert_equal(len(bnb.classes_), 3)
    var preds = bnb.predict(X)
    for i in range(6):
        assert_equal(preds[i], Int(y[i]))


def test_bernoulli_nb_binarization() raises:
    # Continuous values thresholded at 2.5
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 5.0
    X[1, 0] = 4.0
    X[1, 1] = 1.0
    X[2, 0] = 0.5
    X[2, 1] = 3.0
    X[3, 0] = 6.0
    X[3, 1] = 2.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var bnb = BernoulliNB[DType.float64](alpha=1.0, binarize=2.5)
    bnb.fit(X, y)

    var preds = bnb.predict(X)
    for i in range(4):
        assert_equal(preds[i], Int(y[i]))


def test_bernoulli_nb_sparse_csr() raises:
    var indptr = List[Int]()
    indptr.append(0)
    indptr.append(2)
    indptr.append(4)
    indptr.append(7)
    indptr.append(9)
    var indices = List[Int]()
    indices.append(0)
    indices.append(2)
    indices.append(1)
    indices.append(3)
    indices.append(0)
    indices.append(1)
    indices.append(2)
    indices.append(2)
    indices.append(3)
    var data = List[Scalar[DType.float64]]()
    for _ in range(9):
        data.append(1.0)

    var csr = CSRMatrix[DType.float64](4, 4, data^, indices^, indptr^)
    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var bnb = BernoulliNB[DType.float64](alpha=1.0)
    bnb.fit(csr, y)

    var preds = bnb.predict(csr)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 1)
    assert_equal(preds[2], 0)
    assert_equal(preds[3], 1)


def test_bernoulli_nb_uniform_vs_learned_prior() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 1.0
    X[1, 1] = 0.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[3, 0] = 0.0
    X[3, 1] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(0)
    y.append(1)

    var bnb_uniform = BernoulliNB[DType.float64](fit_prior=False)
    bnb_uniform.fit(X, y)
    assert_almost_equal(
        bnb_uniform.class_log_prior_[0], bnb_uniform.class_log_prior_[1]
    )

    var bnb_learned = BernoulliNB[DType.float64](fit_prior=True)
    bnb_learned.fit(X, y)
    assert_true(
        bnb_learned.class_log_prior_[0] > bnb_learned.class_log_prior_[1]
    )


def test_bernoulli_nb_serialization() raises:
    var X = Matrix[DType.float64](4, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[0, 2] = 1.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[1, 2] = 0.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[2, 2] = 1.0
    X[3, 0] = 0.0
    X[3, 1] = 1.0
    X[3, 2] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var bnb = BernoulliNB[DType.float64](alpha=0.5)
    bnb.fit(X, y)

    var bytes = dumps(bnb)
    var restored = loads[BernoulliNB[DType.float64]](bytes)

    assert_true(restored.is_fitted)
    assert_equal(restored.n_features_in_, bnb.n_features_in_)
    assert_almost_equal(restored.alpha, bnb.alpha)

    var p1 = bnb.predict(X)
    var p2 = restored.predict(X)
    for i in range(4):
        assert_equal(p1[i], p2[i])


def test_bernoulli_nb_copy_semantics() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[3, 0] = 0.0
    X[3, 1] = 0.0
    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var bnb1 = BernoulliNB[DType.float64](alpha=1.0)
    bnb1.fit(X, y)

    var bnb2 = bnb1.copy()
    assert_true(bnb2.is_fitted)
    assert_equal(bnb2.n_features_in_, 2)

    var y2 = List[Scalar[DType.int32]]()
    y2.append(1)
    y2.append(0)
    y2.append(1)
    y2.append(0)
    bnb2.fit(X, y2)

    assert_equal(bnb1.predict(X)[0], 0)
    assert_equal(bnb2.predict(X)[0], 1)


def test_bernoulli_nb_error_handling() raises:
    with assert_raises():
        _ = BernoulliNB[DType.float64](alpha=-1.0)

    var bnb = BernoulliNB[DType.float64]()
    var X = Matrix[DType.float64](2, 2, 0)
    with assert_raises():
        _ = bnb.predict(X)

    var y_single_class = List[Scalar[DType.int32]]()
    y_single_class.append(0)
    y_single_class.append(0)
    with assert_raises():
        bnb.fit(X, y_single_class)


def test_complement_nb_binary_dense() raises:
    var X = Matrix[DType.float64](4, 4, 0)
    X[0, 0] = 2.0
    X[0, 1] = 0.0
    X[0, 2] = 1.0
    X[0, 3] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 3.0
    X[1, 2] = 0.0
    X[1, 3] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[2, 2] = 1.0
    X[2, 3] = 0.0
    X[3, 0] = 0.0
    X[3, 1] = 0.0
    X[3, 2] = 2.0
    X[3, 3] = 3.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var cnb = ComplementNB[DType.float64](alpha=1.0)
    cnb.fit(X, y)

    assert_true(cnb.is_fitted)
    assert_equal(len(cnb.classes_), 2)
    assert_equal(cnb.classes_[0], 0)
    assert_equal(cnb.classes_[1], 1)

    var preds = cnb.predict(X)
    for i in range(4):
        assert_equal(preds[i], Int(y[i]))

    var probs = cnb.predict_proba(X)
    assert_equal(probs.rows, 4)
    assert_equal(probs.cols, 2)
    for i in range(4):
        var row_sum = probs[i, 0] + probs[i, 1]
        assert_almost_equal(row_sum, 1.0, rtol=1e-4)


def test_complement_nb_imbalanced_multiclass() raises:
    # 3 classes with imbalanced representation
    var X = Matrix[DType.float64](7, 3, 0)
    # Class 0 (4 samples): rich in feature 0
    X[0, 0] = 5.0
    X[0, 1] = 0.0
    X[0, 2] = 0.0
    X[1, 0] = 4.0
    X[1, 1] = 0.0
    X[1, 2] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 1.0
    X[2, 2] = 0.0
    X[3, 0] = 4.0
    X[3, 1] = 0.0
    X[3, 2] = 0.0
    # Class 1 (2 samples): rich in feature 1
    X[4, 0] = 0.0
    X[4, 1] = 5.0
    X[4, 2] = 0.0
    X[5, 0] = 1.0
    X[5, 1] = 4.0
    X[5, 2] = 0.0
    # Class 2 (1 sample): rich in feature 2
    X[6, 0] = 0.0
    X[6, 1] = 0.0
    X[6, 2] = 6.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(0)
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)
    y.append(2)

    var cnb = ComplementNB[DType.float64](alpha=1.0)
    cnb.fit(X, y)

    var preds = cnb.predict(X)
    for i in range(7):
        assert_equal(preds[i], Int(y[i]))


def test_complement_nb_norm_option() raises:
    var X = Matrix[DType.float64](4, 4, 0)
    X[0, 0] = 2.0
    X[0, 1] = 0.0
    X[0, 2] = 1.0
    X[0, 3] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 3.0
    X[1, 2] = 0.0
    X[1, 3] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[2, 2] = 1.0
    X[2, 3] = 0.0
    X[3, 0] = 0.0
    X[3, 1] = 0.0
    X[3, 2] = 2.0
    X[3, 3] = 3.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var cnb_norm = ComplementNB[DType.float64](alpha=1.0, norm=True)
    cnb_norm.fit(X, y)

    var preds = cnb_norm.predict(X)
    for i in range(4):
        assert_equal(preds[i], Int(y[i]))


def test_complement_nb_sparse_csr() raises:
    var indptr = List[Int]()
    indptr.append(0)
    indptr.append(2)
    indptr.append(4)
    indptr.append(7)
    indptr.append(9)
    var indices = List[Int]()
    indices.append(0)
    indices.append(2)
    indices.append(1)
    indices.append(3)
    indices.append(0)
    indices.append(1)
    indices.append(2)
    indices.append(2)
    indices.append(3)
    var data = List[Scalar[DType.float64]]()
    data.append(2.0)
    data.append(1.0)
    data.append(3.0)
    data.append(1.0)
    data.append(1.0)
    data.append(1.0)
    data.append(1.0)
    data.append(2.0)
    data.append(3.0)

    var csr = CSRMatrix[DType.float64](4, 4, data^, indices^, indptr^)
    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var cnb = ComplementNB[DType.float64](alpha=1.0)
    cnb.fit(csr, y)

    var preds = cnb.predict(csr)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 1)
    assert_equal(preds[2], 0)
    assert_equal(preds[3], 1)


def test_complement_nb_negative_values_rejection() raises:
    var X = Matrix[DType.float64](2, 2, 0)
    X[0, 0] = -1.0
    X[0, 1] = 2.0
    X[1, 0] = 1.0
    X[1, 1] = 0.0
    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)

    var cnb = ComplementNB[DType.float64]()
    with assert_raises():
        cnb.fit(X, y)


def test_complement_nb_serialization() raises:
    var X = Matrix[DType.float64](4, 3, 0)
    X[0, 0] = 2.0
    X[0, 1] = 0.0
    X[0, 2] = 1.0
    X[1, 0] = 0.0
    X[1, 1] = 3.0
    X[1, 2] = 0.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[2, 2] = 2.0
    X[3, 0] = 0.0
    X[3, 1] = 2.0
    X[3, 2] = 1.0

    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var cnb = ComplementNB[DType.float64](alpha=0.5, norm=False)
    cnb.fit(X, y)

    var bytes = dumps(cnb)
    var restored = loads[ComplementNB[DType.float64]](bytes)

    assert_true(restored.is_fitted)
    assert_equal(restored.n_features_in_, cnb.n_features_in_)
    assert_almost_equal(restored.alpha, cnb.alpha)

    var p1 = cnb.predict(X)
    var p2 = restored.predict(X)
    for i in range(4):
        assert_equal(p1[i], p2[i])


def test_complement_nb_copy_semantics() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 2.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 3.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[3, 0] = 0.0
    X[3, 1] = 1.0
    var y = List[Scalar[DType.int32]]()
    y.append(0)
    y.append(1)
    y.append(0)
    y.append(1)

    var cnb1 = ComplementNB[DType.float64](alpha=1.0)
    cnb1.fit(X, y)

    var cnb2 = cnb1.copy()
    assert_true(cnb2.is_fitted)
    assert_equal(cnb2.n_features_in_, 2)

    var y2 = List[Scalar[DType.int32]]()
    y2.append(1)
    y2.append(0)
    y2.append(1)
    y2.append(0)
    cnb2.fit(X, y2)

    assert_equal(cnb1.predict(X)[0], 0)
    assert_equal(cnb2.predict(X)[0], 1)


def test_complement_nb_error_handling() raises:
    with assert_raises():
        _ = ComplementNB[DType.float64](alpha=-0.5)

    var cnb = ComplementNB[DType.float64]()
    var X = Matrix[DType.float64](2, 2, 0)
    with assert_raises():
        _ = cnb.predict(X)

    var y_single_class = List[Scalar[DType.int32]]()
    y_single_class.append(0)
    y_single_class.append(0)
    with assert_raises():
        cnb.fit(X, y_single_class)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
