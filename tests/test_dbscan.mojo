from std.math import nan, inf, cos, sin
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
from strata.cluster.dbscan import DBSCAN
from strata.exceptions.errors import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_dbscan_basic_two_clusters_with_noise() raises:
    # 3 points around (0, 0), 3 points around (10, 10), 1 outlier at (50, 50)
    var X = Matrix[DType.float64](7, 2, 0)
    # Cluster A
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 0.1

    # Cluster B
    X[3, 0] = 10.0
    X[3, 1] = 10.0
    X[4, 0] = 10.1
    X[4, 1] = 10.0
    X[5, 0] = 10.0
    X[5, 1] = 10.1

    # Noise point
    X[6, 0] = 50.0
    X[6, 1] = 50.0

    var db = DBSCAN[DType.float64](eps=0.5, min_samples=3)
    db.fit(X)

    assert_true(db.is_fitted)
    assert_equal(db.n_clusters_, 2)
    assert_equal(len(db.labels_), 7)

    # First 3 belong to cluster 0
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 0)

    # Next 3 belong to cluster 1
    assert_equal(db.labels_[3], 1)
    assert_equal(db.labels_[4], 1)
    assert_equal(db.labels_[5], 1)

    # Last point is noise (-1)
    assert_equal(db.labels_[6], -1)

    # Points 0, 1, 2, 3, 4, 5 are core samples (each has 3 neighbors within 0.5)
    assert_equal(len(db.core_sample_indices_), 6)
    assert_equal(db.components_.rows, 6)
    assert_equal(db.components_.cols, 2)


def test_dbscan_border_point_identification() raises:
    # 3 points forming a core at (0, 0), and 1 border point at (0.4, 0.0)
    # min_samples = 3.
    # Points 0, 1, 2 have each other + border point -> >= 3 neighbors -> core.
    # Border point 3 only has Point 0 and itself within 0.5 (Point 1 and 2 are slightly further than 0.5)
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 0.3
    X[2, 0] = 0.0
    X[2, 1] = -0.3
    # Point 3 is distance 0.4 from Point 0, but distance sqrt(0.4^2 + 0.3^2) = 0.5 from Point 1 and 2
    # Let's place it at (0.42, 0.0) so distance to 1 and 2 is sqrt(0.42^2 + 0.09) = 0.516 > 0.5
    X[3, 0] = 0.42
    X[3, 1] = 0.0

    var db = DBSCAN[DType.float64](eps=0.5, min_samples=3)
    db.fit(X)

    assert_equal(db.n_clusters_, 1)
    # All 4 points must belong to cluster 0 (point 3 is a border point)
    for i in range(4):
        assert_equal(db.labels_[i], 0)

    # But core_sample_indices_ must NOT include point 3!
    var has_point_3 = False
    for i in range(len(db.core_sample_indices_)):
        if db.core_sample_indices_[i] == 3:
            has_point_3 = True
    assert_false(has_point_3)
    # Point 0 has neighbors {0, 1, 2, 3} (4 points >= 3) -> Core
    var has_point_0 = False
    for i in range(len(db.core_sample_indices_)):
        if db.core_sample_indices_[i] == 0:
            has_point_0 = True
    assert_true(has_point_0)


def test_dbscan_all_noise() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 10.0
    X[1, 1] = 10.0
    X[2, 0] = 20.0
    X[2, 1] = 20.0
    X[3, 0] = 30.0
    X[3, 1] = 30.0

    var db = DBSCAN[DType.float64](eps=1.0, min_samples=2)
    db.fit(X)

    assert_equal(db.n_clusters_, 0)
    assert_equal(len(db.core_sample_indices_), 0)
    for i in range(4):
        assert_equal(db.labels_[i], -1)


def test_dbscan_single_cluster() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.5
    X[1, 1] = 0.0
    X[2, 0] = 1.0
    X[2, 1] = 0.0
    X[3, 0] = 1.5
    X[3, 1] = 0.0

    var db = DBSCAN[DType.float64](eps=0.8, min_samples=2)
    var labels = db.fit_predict(X)

    assert_equal(db.n_clusters_, 1)
    for i in range(4):
        assert_equal(labels[i], 0)
        assert_equal(db.labels_[i], 0)


def test_dbscan_single_sample() raises:
    var X = Matrix[DType.float64](1, 2, 0)
    X[0, 0] = 3.14
    X[0, 1] = 2.71

    # With min_samples=1, 1 point is its own core and forms a cluster
    var db1 = DBSCAN[DType.float64](eps=1.0, min_samples=1)
    db1.fit(X)
    assert_equal(db1.n_clusters_, 1)
    assert_equal(db1.labels_[0], 0)
    assert_equal(len(db1.core_sample_indices_), 1)

    # With min_samples=2, 1 point cannot be core and becomes noise
    var db2 = DBSCAN[DType.float64](eps=1.0, min_samples=2)
    db2.fit(X)
    assert_equal(db2.n_clusters_, 0)
    assert_equal(db2.labels_[0], -1)
    assert_equal(len(db2.core_sample_indices_), 0)


def test_dbscan_duplicate_points() raises:
    # 5 identical points at (2.0, 3.0) and 1 distinct point at (2.1, 3.1)
    var X = Matrix[DType.float64](6, 2, 0)
    for i in range(5):
        X[i, 0] = 2.0
        X[i, 1] = 3.0
    X[5, 0] = 2.1
    X[5, 1] = 3.1

    var db = DBSCAN[DType.float64](eps=0.5, min_samples=3)
    db.fit(X)

    assert_equal(db.n_clusters_, 1)
    for i in range(6):
        assert_equal(db.labels_[i], 0)
    assert_equal(len(db.core_sample_indices_), 6)


def test_dbscan_non_convex_concentric_rings() raises:
    # Inner ring (radius 1.0, 8 points) and Outer ring (radius 5.0, 12 points)
    # Total 20 points in 2 dimensions
    var n_inner = 8
    var n_outer = 12
    var total_pts = n_inner + n_outer
    var X = Matrix[DType.float64](total_pts, 2, 0)

    # Inner ring: radius 1.0. Arc distance between adjacent points is 2*pi*1 / 8 = 0.785
    for i in range(n_inner):
        var theta = Float64(i) * 6.283185307179586 / Float64(n_inner)
        X[i, 0] = cos(theta) * 1.0
        X[i, 1] = sin(theta) * 1.0

    # Outer ring: radius 5.0. Distance to inner ring is 4.0 >> eps
    # Arc distance between outer points: 2*pi*5 / 12 = 2.618
    for i in range(n_outer):
        var theta = Float64(i) * 6.283185307179586 / Float64(n_outer)
        X[n_inner + i, 0] = cos(theta) * 5.0
        X[n_inner + i, 1] = sin(theta) * 5.0

    # eps=1.0 connects points in inner ring (chord dist ~ 0.765 <= 1.0),
    # but does not connect outer points (chord dist ~ 2.58 > 1.0) nor outer to inner (dist ~ 4.0)
    var db_inner = DBSCAN[DType.float64](eps=1.0, min_samples=2)
    db_inner.fit(X)

    assert_equal(db_inner.n_clusters_, 1)
    # Inner ring points belong to cluster 0
    for i in range(n_inner):
        assert_equal(db_inner.labels_[i], 0)
    # Outer ring points are all noise (-1)
    for i in range(n_inner, total_pts):
        assert_equal(db_inner.labels_[i], -1)

    # Now with eps=3.0, both rings form separate clusters!
    var db_both = DBSCAN[DType.float64](eps=3.0, min_samples=2)
    db_both.fit(X)

    assert_equal(db_both.n_clusters_, 2)
    var inner_label = db_both.labels_[0]
    var outer_label = db_both.labels_[n_inner]
    assert_true(inner_label != outer_label)
    assert_true(inner_label >= 0)
    assert_true(outer_label >= 0)

    for i in range(n_inner):
        assert_equal(db_both.labels_[i], inner_label)
    for i in range(n_inner, total_pts):
        assert_equal(db_both.labels_[i], outer_label)


def test_dbscan_high_dimensional_data() raises:
    # 6 points in 10-dimensional space
    var X = Matrix[DType.float64](6, 10, 0)
    # Cluster 0: points 0, 1, 2
    for j in range(10):
        X[0, j] = 1.0
        X[1, j] = 1.05
        X[2, j] = 0.95
    # Cluster 1: points 3, 4, 5
    for j in range(10):
        X[3, j] = 50.0
        X[4, j] = 50.05
        X[5, j] = 49.95

    var db = DBSCAN[DType.float64](eps=1.0, min_samples=2)
    db.fit(X)

    assert_equal(db.n_clusters_, 2)
    assert_equal(db.n_features_in_, 10)
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 0)
    assert_equal(db.labels_[3], 1)
    assert_equal(db.labels_[4], 1)
    assert_equal(db.labels_[5], 1)


def test_dbscan_fit_predict_consistency() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.1
    X[1, 1] = 1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.2
    X[3, 0] = 2.0
    X[3, 1] = 2.0
    X[4, 0] = 100.0
    X[4, 1] = 100.0

    var db1 = DBSCAN[DType.float64](eps=0.5, min_samples=2)
    db1.fit(X)

    var db2 = DBSCAN[DType.float64](eps=0.5, min_samples=2)
    var pred_labels = db2.fit_predict(X)

    assert_equal(len(db1.labels_), len(pred_labels))
    for i in range(len(db1.labels_)):
        assert_equal(db1.labels_[i], pred_labels[i])


def test_dbscan_predict_out_of_sample() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 0.1

    X[3, 0] = 10.0
    X[3, 1] = 10.0
    X[4, 0] = 10.1
    X[4, 1] = 10.0
    X[5, 0] = 10.0
    X[5, 1] = 10.1

    var db = DBSCAN[DType.float64](eps=0.5, min_samples=3)
    db.fit(X)

    var X_query = Matrix[DType.float64](3, 2, 0)
    # Query 0 near Cluster 0
    X_query[0, 0] = 0.05
    X_query[0, 1] = 0.05
    # Query 1 near Cluster 1
    X_query[1, 0] = 10.05
    X_query[1, 1] = 10.05
    # Query 2 out in the middle of nowhere
    X_query[2, 0] = 5.0
    X_query[2, 1] = 5.0

    var preds = db.predict(X_query)
    assert_equal(len(preds), 3)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 1)
    assert_equal(preds[2], -1)


def test_dbscan_predict_empty_queries() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    for i in range(4):
        X[i, 0] = Float64(i) * 0.1
        X[i, 1] = 0.0
    var db = DBSCAN[DType.float64](eps=0.5, min_samples=2)
    db.fit(X)

    var empty_query = Matrix[DType.float64](0, 2, 0)
    with assert_raises():
        _ = db.predict(empty_query)


def test_dbscan_predict_when_all_noise() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 10.0
    X[1, 1] = 10.0
    X[2, 0] = 20.0
    X[2, 1] = 20.0

    var db = DBSCAN[DType.float64](eps=0.5, min_samples=2)
    db.fit(X)
    assert_equal(db.n_clusters_, 0)

    var X_query = Matrix[DType.float64](2, 2, 0)
    var preds = db.predict(X_query)
    assert_equal(len(preds), 2)
    assert_equal(preds[0], -1)
    assert_equal(preds[1], -1)


def test_dbscan_brute_vs_kdtree_parity() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.2
    X[1, 1] = 1.1
    X[2, 0] = 1.1
    X[2, 1] = 1.3
    X[3, 0] = 5.0
    X[3, 1] = 5.0
    X[4, 0] = 5.1
    X[4, 1] = 5.2
    X[5, 0] = 20.0
    X[5, 1] = 20.0

    var db_kd = DBSCAN[DType.float64](
        eps=0.6, min_samples=2, algorithm="kd_tree"
    )
    db_kd.fit(X)

    var db_brute = DBSCAN[DType.float64](
        eps=0.6, min_samples=2, algorithm="brute"
    )
    db_brute.fit(X)

    assert_equal(db_kd.n_clusters_, db_brute.n_clusters_)
    assert_equal(len(db_kd.labels_), len(db_brute.labels_))
    for i in range(len(db_kd.labels_)):
        assert_equal(db_kd.labels_[i], db_brute.labels_[i])


def test_dbscan_manhattan_metric() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.5
    X[1, 1] = 0.4  # Manhattan dist = 0.9
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 5.4
    X[3, 1] = 5.5  # Manhattan dist = 0.9

    var db = DBSCAN[DType.float64](eps=1.0, min_samples=2, metric="manhattan")
    db.fit(X)

    assert_equal(db.n_clusters_, 2)
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 1)
    assert_equal(db.labels_[3], 1)


def test_dbscan_cityblock_metric() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.3
    X[1, 1] = 0.3  # Cityblock = 0.6 <= 0.8
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.4
    X[3, 1] = 10.3  # Cityblock = 0.7 <= 0.8

    var db = DBSCAN[DType.float64](eps=0.8, min_samples=2, metric="cityblock")
    db.fit(X)

    assert_equal(db.n_clusters_, 2)
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 1)
    assert_equal(db.labels_[3], 1)


def test_dbscan_chebyshev_metric() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.8
    X[1, 1] = 0.2  # Chebyshev = max(0.8, 0.2) = 0.8 <= 0.9
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.8
    X[3, 1] = 10.1  # Chebyshev = max(0.8, 0.1) = 0.8 <= 0.9

    var db = DBSCAN[DType.float64](eps=0.9, min_samples=2, metric="chebyshev")
    db.fit(X)

    assert_equal(db.n_clusters_, 2)
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 1)
    assert_equal(db.labels_[3], 1)


def test_dbscan_cosine_metric_and_predict() raises:
    # Cosine distance: 1 - (u . v) / (||u|| * ||v||)
    # Vectors with identical angle have cosine distance 0
    var X = Matrix[DType.float64](4, 2, 0)
    # Pair 1 along direction (1, 1)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 3.0
    X[1, 1] = 3.0  # Same direction, dist = 0.0
    # Pair 2 along direction (1, -1)
    X[2, 0] = 2.0
    X[2, 1] = -2.0
    X[3, 0] = 5.0
    X[3, 1] = -5.0  # Same direction, dist = 0.0

    var db = DBSCAN[DType.float64](
        eps=0.1, min_samples=2, metric="cosine", algorithm="brute"
    )
    db.fit(X)

    assert_equal(db.n_clusters_, 2)
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 1)
    assert_equal(db.labels_[3], 1)

    # Test predict with cosine metric
    var X_q = Matrix[DType.float64](2, 2, 0)
    X_q[0, 0] = 10.0
    X_q[0, 1] = 10.0  # Direction (1, 1) -> Cluster 0
    X_q[1, 0] = 10.0
    X_q[1, 1] = -10.0  # Direction (1, -1) -> Cluster 1

    var preds = db.predict(X_q)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 1)


def test_dbscan_minkowski_metric() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.5
    X[1, 1] = 0.5  # p=3 distance = (0.5^3 + 0.5^3)^(1/3) = 0.6299 <= 0.7
    X[2, 0] = 20.0
    X[2, 1] = 20.0
    X[3, 0] = 20.5
    X[3, 1] = 20.5

    var db = DBSCAN[DType.float64](
        eps=0.7, min_samples=2, metric="minkowski", p=3.0, algorithm="brute"
    )
    db.fit(X)

    assert_equal(db.n_clusters_, 2)
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 1)
    assert_equal(db.labels_[3], 1)


def test_dbscan_dataset_interop() raises:
    var records = Matrix[DType.float64](4, 2, 0)
    records[0, 0] = 0.0
    records[0, 1] = 0.0
    records[1, 0] = 0.1
    records[1, 1] = 0.1
    records[2, 0] = 10.0
    records[2, 1] = 10.0
    records[3, 0] = 10.1
    records[3, 1] = 10.1

    var targets = List[Scalar[DType.float64]](capacity=4)
    for _ in range(4):
        targets.append(0.0)

    var ds = Dataset[DType.float64, DType.float64](records^, targets^)
    var db = DBSCAN[DType.float64](eps=0.5, min_samples=2)

    # Use generic fit and predict free functions for Dataset containers
    fit_dataset[DBSCAN[DType.float64], DType.float64, DType.float64](db, ds)
    assert_true(db.is_fitted)
    assert_equal(db.n_clusters_, 2)

    var preds = predict_dataset[
        DBSCAN[DType.float64], DType.float64, DType.float64
    ](db, ds)
    assert_equal(len(preds), 4)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_dbscan_nan_inf_rejection() raises:
    var X_nan = Matrix[DType.float64](3, 2, 0)
    X_nan[0, 0] = nan[DType.float64]()
    var db = DBSCAN[DType.float64]()
    with assert_raises():
        db.fit(X_nan)

    var X_inf = Matrix[DType.float64](3, 2, 0)
    X_inf[0, 0] = inf[DType.float64]()
    with assert_raises():
        db.fit(X_inf)


def test_dbscan_parameter_validation() raises:
    # Invalid eps
    with assert_raises():
        _ = DBSCAN[DType.float64](eps=0.0)
    with assert_raises():
        _ = DBSCAN[DType.float64](eps=-1.0)

    # Invalid min_samples
    with assert_raises():
        _ = DBSCAN[DType.float64](min_samples=0)

    # Invalid algorithm
    with assert_raises():
        _ = DBSCAN[DType.float64](algorithm="fast_cluster")

    # Invalid metric for KDTree
    with assert_raises():
        _ = DBSCAN[DType.float64](algorithm="kd_tree", metric="cosine")

    # NotFittedError on predict
    var unfitted_db = DBSCAN[DType.float64]()
    var dummy = Matrix[DType.float64](2, 2, 0)
    with assert_raises():
        _ = unfitted_db.predict(dummy)

    # DimensionMismatchError on fit
    var empty_X = Matrix[DType.float64](0, 0, 0)
    var db = DBSCAN[DType.float64]()
    with assert_raises():
        db.fit(empty_X)

    # DimensionMismatchError on predict
    var X_train = Matrix[DType.float64](3, 2, 0)
    db.fit(X_train)
    var X_wrong_dim = Matrix[DType.float64](2, 3, 0)
    with assert_raises():
        _ = db.predict(X_wrong_dim)


def test_dbscan_repeated_fit_resets_state() raises:
    var X1 = Matrix[DType.float64](4, 2, 0)
    X1[0, 0] = 0.0
    X1[0, 1] = 0.0
    X1[1, 0] = 0.1
    X1[1, 1] = 0.1
    X1[2, 0] = 10.0
    X1[2, 1] = 10.0
    X1[3, 0] = 10.1
    X1[3, 1] = 10.1

    var db = DBSCAN[DType.float64](eps=0.5, min_samples=2)
    db.fit(X1)
    assert_equal(db.n_clusters_, 2)
    assert_equal(len(db.labels_), 4)

    # Fit again with 2 distant points
    var X2 = Matrix[DType.float64](2, 2, 0)
    X2[0, 0] = 0.0
    X2[0, 1] = 0.0
    X2[1, 0] = 100.0
    X2[1, 1] = 100.0
    db.fit(X2)
    assert_equal(db.n_clusters_, 0)
    assert_equal(len(db.labels_), 2)
    assert_equal(db.labels_[0], -1)
    assert_equal(db.labels_[1], -1)


def test_dbscan_float32() raises:
    var X = Matrix[DType.float32](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.1
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.1
    X[3, 1] = 10.1

    var db = DBSCAN[DType.float32](eps=0.5, min_samples=2)
    db.fit(X)

    assert_equal(db.n_clusters_, 2)
    assert_equal(db.labels_[0], 0)
    assert_equal(db.labels_[1], 0)
    assert_equal(db.labels_[2], 1)
    assert_equal(db.labels_[3], 1)


def test_dbscan_copy() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.1
    X[1, 1] = 0.1
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.1
    X[3, 1] = 10.1

    var db = DBSCAN[DType.float64](eps=0.5, min_samples=2)
    db.fit(X)

    var db_copy = DBSCAN(copy=db)
    assert_equal(db_copy.n_clusters_, db.n_clusters_)
    assert_equal(len(db_copy.labels_), len(db.labels_))
    for i in range(len(db.labels_)):
        assert_equal(db_copy.labels_[i], db.labels_[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
