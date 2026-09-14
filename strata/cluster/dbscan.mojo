from std.math import sqrt
from ..base.estimator import Clusterer
from ..core.matrix import Matrix
from ..utils.validation import (
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..exceptions.errors import (
    InvalidParameterError,
    DimensionMismatchError,
    NotFittedError,
)
from ..neighbors.kd_tree import KDTree
from ..neighbors.distance import (
    row_distance,
    euclidean_distance,
    pairwise_distances,
    _validate_metric_and_p,
)


struct DBSCAN[compute_dtype: DType = DType.float64](
    Clusterer, Copyable, Movable
):
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Finds core samples in regions of high density and expands clusters from them.
    Suitable for spatial clustering with arbitrary non-convex geometries and
    identifies outliers as noise (label -1).

    Parameters:
        compute_dtype: Precision for spatial coordinate computation. Default DType.float64.

    Args:
        eps: The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. Default 0.5.
        min_samples: The number of samples in a neighborhood for a point to be
            considered as a core point (including the point itself). Default 5.
        metric: The distance metric to use ('euclidean', 'l2', 'manhattan', 'l1',
            'chebyshev', 'minkowski', 'cosine'). Default 'euclidean'.
        algorithm: Algorithm used to compute nearest neighborhoods ('auto', 'kd_tree', 'brute').
            Default 'auto'.
        leaf_size: Leaf size passed to KDTree when applicable. Default 30.
        p: The power of the Minkowski metric when metric='minkowski'. Default 2.0.

    Attributes:
        core_sample_indices_: Indices of core samples.
        components_: Coordinates of each core sample of shape (n_core_samples, n_features).
        labels_: Cluster labels for each point in dataset given to fit(). Noisy samples are -1.
        n_features_in_: Number of features seen during fit.
        n_clusters_: Number of clusters found (excluding noise).
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.cluster import DBSCAN
        from strata.core import Matrix

        var X = Matrix[DType.float64](6, 2)
        X[0, 0] = 1.0; X[0, 1] = 2.0
        X[1, 0] = 2.0; X[1, 1] = 2.0
        X[2, 0] = 2.0; X[2, 1] = 3.0
        X[3, 0] = 8.0; X[3, 1] = 7.0
        X[4, 0] = 8.0; X[4, 1] = 8.0
        X[5, 0] = 25.0; X[5, 1] = 80.0

        var db = DBSCAN(eps=3.0, min_samples=2)
        db.fit(X)
        print("Labels:", db.labels_[0], db.labels_[3], db.labels_[5])
        # Cluster 0, Cluster 1, Noise (-1)
        ```
    """

    var eps: Float64
    var min_samples: Int
    var metric: String
    var algorithm: String
    var leaf_size: Int
    var p: Float64

    var core_sample_indices_: List[Int]
    var components_: Matrix[Self.compute_dtype]
    var labels_: List[Int]
    var n_features_in_: Int
    var n_clusters_: Int
    var is_fitted: Bool

    def __init__(
        out self,
        eps: Float64 = 0.5,
        min_samples: Int = 5,
        metric: String = "euclidean",
        algorithm: String = "auto",
        leaf_size: Int = 30,
        p: Float64 = 2.0,
    ) raises:
        """Initialize DBSCAN with hyperparameters."""
        check_floating_dtype[Self.compute_dtype, "DBSCAN"]()

        if eps <= 0.0:
            raise InvalidParameterError.error(
                "eps", "eps must be > 0.0, got " + String(eps)
            )
        if min_samples < 1:
            raise InvalidParameterError.error(
                "min_samples",
                "min_samples must be >= 1, got " + String(min_samples),
            )
        if (
            algorithm != "auto"
            and algorithm != "kd_tree"
            and algorithm != "brute"
        ):
            raise InvalidParameterError.error(
                "algorithm",
                "algorithm must be 'auto', 'kd_tree', or 'brute', got '"
                + algorithm
                + "'",
            )
        _validate_metric_and_p(metric, p)
        if algorithm == "kd_tree":
            if (
                metric != "euclidean"
                and metric != "l2"
                and metric != "manhattan"
                and metric != "cityblock"
                and metric != "l1"
                and metric != "chebyshev"
            ):
                raise InvalidParameterError.error(
                    "metric",
                    "metric '"
                    + metric
                    + "' is not supported by KDTree. Use algorithm='brute' or"
                    " 'auto'.",
                )

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p

        self.core_sample_indices_ = List[Int]()
        self.components_ = Matrix[Self.compute_dtype](0, 0)
        self.labels_ = List[Int]()
        self.n_features_in_ = 0
        self.n_clusters_ = 0
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        """Copy constructor."""
        self.eps = copy.eps
        self.min_samples = copy.min_samples
        self.metric = copy.metric
        self.algorithm = copy.algorithm
        self.leaf_size = copy.leaf_size
        self.p = copy.p

        self.core_sample_indices_ = copy.core_sample_indices_.copy()
        self.components_ = copy.components_.copy()
        self.labels_ = copy.labels_.copy()
        self.n_features_in_ = copy.n_features_in_
        self.n_clusters_ = copy.n_clusters_
        self.is_fitted = copy.is_fitted

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Perform DBSCAN clustering from feature matrix X.

        Args:
            X: Input matrix with shape (n_samples, n_features).

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or Inf.
        """
        check_array[in_dtype](X)
        if X.rows == 0 or X.cols == 0:
            raise DimensionMismatchError.error(
                "X.rows > 0 and X.cols > 0",
                "X.rows == " + String(X.rows) + ", X.cols == " + String(X.cols),
                "DBSCAN.fit",
            )

        var n = X.rows
        var d = X.cols
        self.n_features_in_ = d

        var X_comp = X.cast[Self.compute_dtype]()

        # Retrieve radius neighborhoods for each point
        var neighborhoods = List[List[Int]](capacity=n)
        var use_kdtree = self.algorithm == "kd_tree" or (
            self.algorithm == "auto"
            and (
                self.metric == "euclidean"
                or self.metric == "l2"
                or self.metric == "manhattan"
                or self.metric == "cityblock"
                or self.metric == "l1"
                or self.metric == "chebyshev"
            )
        )

        if use_kdtree:
            var tree = KDTree[Self.compute_dtype](X_comp, metric=self.metric)
            var query_res = tree.query_radius[Self.compute_dtype](
                X_comp, self.eps
            )
            neighborhoods = query_res[1].copy()
        else:
            var D = pairwise_distances[Self.compute_dtype](
                X_comp, metric=self.metric, p=self.p
            )
            for i in range(n):
                var nbrs = List[Int]()
                for j in range(n):
                    if Float64(D[i, j]) <= self.eps:
                        nbrs.append(j)
                neighborhoods.append(nbrs^)

        # Distinguish states: UNASSIGNED = -2, NOISE = -1, clusters >= 0
        comptime UNASSIGNED = -2
        comptime NOISE = -1

        var labels = List[Int](capacity=n)
        for _ in range(n):
            labels.append(UNASSIGNED)

        var is_core = List[Bool](capacity=n)
        var core_sample_indices = List[Int]()
        for i in range(n):
            if len(neighborhoods[i]) >= self.min_samples:
                is_core.append(True)
                core_sample_indices.append(i)
            else:
                is_core.append(False)

        var cluster_id = 0
        for i in range(n):
            if labels[i] != UNASSIGNED:
                continue

            if not is_core[i]:
                labels[i] = NOISE
                continue

            # Core point found: create new cluster
            labels[i] = cluster_id

            var in_queue = List[Bool](capacity=n)
            for _ in range(n):
                in_queue.append(False)
            in_queue[i] = True

            var queue = List[Int]()
            for j in range(len(neighborhoods[i])):
                var neighbor = neighborhoods[i][j]
                if neighbor != i:
                    queue.append(neighbor)
                    in_queue[neighbor] = True

            var head = 0
            while head < len(queue):
                var current = queue[head]
                head += 1

                # If previously marked noise, point was reached by a core point -> border point
                if labels[current] == NOISE:
                    labels[current] = cluster_id

                if labels[current] != UNASSIGNED:
                    continue

                labels[current] = cluster_id

                # If current point is also core, expand the frontier
                if is_core[current]:
                    for j in range(len(neighborhoods[current])):
                        var neighbor = neighborhoods[current][j]
                        if not in_queue[neighbor]:
                            in_queue[neighbor] = True
                            queue.append(neighbor)

            cluster_id += 1

        for i in range(n):
            if labels[i] == UNASSIGNED:
                labels[i] = NOISE

        # Build components_ matrix holding all core sample coordinates
        var num_core = len(core_sample_indices)
        var components = Matrix[Self.compute_dtype](num_core, d)
        for k in range(num_core):
            var orig_idx = core_sample_indices[k]
            for j in range(d):
                components[k, j] = X_comp[orig_idx, j]

        self.core_sample_indices_ = core_sample_indices^
        self.components_ = components^
        self.labels_ = labels^
        self.n_clusters_ = cluster_id
        self.is_fitted = True

    def fit_predict[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> List[Int]:
        """Compute clusters from X and return cluster labels.

        Args:
            X: Input matrix with shape (n_samples, n_features).

        Returns:
            List[Int]: Cluster labels for each point in X. Noise is -1.
        """
        self.fit[in_dtype](X)
        return self.labels_.copy()

    def predict[in_dtype: DType](self, X: Matrix[in_dtype]) raises -> List[Int]:
        """Predict cluster labels for new points based on nearest core sample.

        Assigns each query sample to the cluster of the nearest core sample if
        within distance <= eps; otherwise labels as noise (-1).

        Args:
            X: Query matrix with shape (n_queries, n_features).

        Returns:
            List[Int]: Predicted cluster labels (noise is -1).

        Raises:
            NotFittedError: If DBSCAN has not been fitted.
            DimensionMismatchError: If X.cols != n_features_in_.
        """
        check_is_fitted("DBSCAN", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "DBSCAN.predict",
            )

        var n_queries = X.rows
        var preds = List[Int](capacity=n_queries)
        if len(self.core_sample_indices_) == 0 or n_queries == 0:
            for _ in range(n_queries):
                preds.append(-1)
            return preds^

        var X_comp = X.cast[Self.compute_dtype]()

        # Map each core sample to its cluster label
        var core_labels = List[Int](capacity=len(self.core_sample_indices_))
        for idx in range(len(self.core_sample_indices_)):
            core_labels.append(self.labels_[self.core_sample_indices_[idx]])

        var use_kdtree = self.algorithm == "kd_tree" or (
            self.algorithm == "auto"
            and (
                self.metric == "euclidean"
                or self.metric == "l2"
                or self.metric == "manhattan"
                or self.metric == "cityblock"
                or self.metric == "l1"
                or self.metric == "chebyshev"
            )
        )

        if use_kdtree:
            var tree = KDTree[Self.compute_dtype](
                self.components_, metric=self.metric
            )
            var res = tree.query[Self.compute_dtype](X_comp, k=1)
            var dists = res[0].copy()
            var indices = res[1].copy()
            for q in range(n_queries):
                var d_val = Float64(dists[q, 0])
                if d_val <= self.eps:
                    preds.append(core_labels[Int(indices[q, 0])])
                else:
                    preds.append(-1)
        else:
            for q in range(n_queries):
                var min_d: Float64 = 1e300
                var min_idx: Int = -1
                for c in range(self.components_.rows):
                    var d_val = Float64(
                        row_distance[Self.compute_dtype](
                            X_comp, q, self.components_, c, self.metric, self.p
                        )
                    )
                    if d_val < min_d:
                        min_d = d_val
                        min_idx = c
                if min_d <= self.eps and min_idx >= 0:
                    preds.append(core_labels[min_idx])
                else:
                    preds.append(-1)

        return preds^
