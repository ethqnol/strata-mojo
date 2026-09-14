from std.math import sqrt
from std.time import perf_counter_ns
from strata import Matrix, CSRMatrix
from strata.utils.random import PRNG


struct BenchResult(Copyable, Movable):
    var benchmark: String
    var phase: String
    var samples: Int
    var features: Int
    var median_ms: Float64
    var mean_ms: Float64
    var min_ms: Float64
    var max_ms: Float64
    var std_ms: Float64
    var throughput_samples_per_sec: Float64
    var metric_name: String
    var metric_val: Float64
    var iterations: Int

    def __init__(
        out self,
        benchmark: String,
        phase: String,
        samples: Int,
        features: Int,
        median_ms: Float64,
        mean_ms: Float64,
        min_ms: Float64,
        max_ms: Float64,
        std_ms: Float64,
        throughput_samples_per_sec: Float64,
        metric_name: String,
        metric_val: Float64,
        iterations: Int,
    ):
        self.benchmark = benchmark
        self.phase = phase
        self.samples = samples
        self.features = features
        self.median_ms = median_ms
        self.mean_ms = mean_ms
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.std_ms = std_ms
        self.throughput_samples_per_sec = throughput_samples_per_sec
        self.metric_name = metric_name
        self.metric_val = metric_val
        self.iterations = iterations

    def to_json(self) -> String:
        var res = String('{"benchmark":"') + self.benchmark + '"'
        res += ',"phase":"' + self.phase + '"'
        res += ',"samples":' + String(self.samples)
        res += ',"features":' + String(self.features)
        res += ',"median_ms":' + String(self.median_ms)
        res += ',"mean_ms":' + String(self.mean_ms)
        res += ',"min_ms":' + String(self.min_ms)
        res += ',"max_ms":' + String(self.max_ms)
        res += ',"std_ms":' + String(self.std_ms)
        res += ',"throughput_samples_per_sec":' + String(
            self.throughput_samples_per_sec
        )
        res += ',"metric_name":"' + self.metric_name + '"'
        res += ',"metric_val":' + String(self.metric_val)
        res += ',"iterations":' + String(self.iterations) + "}"
        return res


struct BenchTimer(Copyable, Movable):
    var times_ns: List[Int]

    def __init__(out self):
        self.times_ns = List[Int]()

    def add(mut self, elapsed_ns: Int):
        self.times_ns.append(elapsed_ns)

    def compute_stats(
        self,
        benchmark: String,
        phase: String,
        samples: Int,
        features: Int,
        metric_name: String = "none",
        metric_val: Float64 = 0.0,
    ) -> BenchResult:
        var n = len(self.times_ns)
        if n == 0:
            return BenchResult(
                benchmark,
                phase,
                samples,
                features,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                metric_name,
                metric_val,
                0,
            )

        var ms_list = List[Float64](capacity=n)
        var sum_ms: Float64 = 0.0
        var min_ms: Float64 = 1e18
        var max_ms: Float64 = 0.0

        for i in range(n):
            var ms = Float64(self.times_ns[i]) / 1000000.0
            ms_list.append(ms)
            sum_ms += ms
            if ms < min_ms:
                min_ms = ms
            if ms > max_ms:
                max_ms = ms

        var mean_ms = sum_ms / Float64(n)

        # In-place insertion sort to find median
        for i in range(1, n):
            var key = ms_list[i]
            var j = i - 1
            while j >= 0 and ms_list[j] > key:
                ms_list[j + 1] = ms_list[j]
                j -= 1
            ms_list[j + 1] = key

        var median_ms: Float64
        if n % 2 == 1:
            median_ms = ms_list[n // 2]
        else:
            median_ms = (ms_list[(n // 2) - 1] + ms_list[n // 2]) / 2.0

        # Variance and standard deviation
        var sum_sq_diff: Float64 = 0.0
        for i in range(n):
            var diff = ms_list[i] - mean_ms
            sum_sq_diff += diff * diff
        var std_ms = sqrt(sum_sq_diff / Float64(n))

        # Throughput: samples per second based on median
        var throughput: Float64 = 0.0
        if median_ms > 0.0:
            throughput = (Float64(samples) / median_ms) * 1000.0

        return BenchResult(
            benchmark,
            phase,
            samples,
            features,
            median_ms,
            mean_ms,
            min_ms,
            max_ms,
            std_ms,
            throughput,
            metric_name,
            metric_val,
            n,
        )


def make_synthetic_regression(
    n_samples: Int, n_features: Int, seed: Int = 42
) -> Tuple[Matrix[DType.float64], List[Scalar[DType.float64]]]:
    """Generates synthetic multi-feature linear regression dataset deterministically.
    """
    var rng = PRNG(seed)
    var X = Matrix[DType.float64](n_samples, n_features, 0.0)
    var weights = List[Float64](capacity=n_features)
    for _ in range(n_features):
        weights.append(Float64(rng.next_int(2000) - 1000) / 100.0)

    var y = List[Scalar[DType.float64]](capacity=n_samples)

    for i in range(n_samples):
        var dot_val: Float64 = 2.5  # true intercept
        for j in range(n_features):
            var val = Float64(rng.next_int(2000) - 1000) / 100.0
            X[i, j] = val
            dot_val += val * weights[j]
        # Add small deterministic noise
        var noise = Float64(rng.next_int(200) - 100) / 500.0
        y.append(dot_val + noise)

    return (X^, y^)


def make_synthetic_classification(
    n_samples: Int, n_features: Int, n_classes: Int = 2, seed: Int = 42
) -> Tuple[Matrix[DType.float64], List[Scalar[DType.int32]]]:
    """Generates synthetic classification dataset with separable clustered features.
    """
    var rng = PRNG(seed)
    var X = Matrix[DType.float64](n_samples, n_features, 0.0)
    var y = List[Scalar[DType.int32]](capacity=n_samples)

    # Class centroids
    var centroids = Matrix[DType.float64](n_classes, n_features, 0.0)
    for c in range(n_classes):
        for j in range(n_features):
            centroids[c, j] = (
                Float64(c * 3) + Float64(rng.next_int(200) - 100) / 100.0
            )

    for i in range(n_samples):
        var c = i % n_classes
        y.append(Int32(c))
        for j in range(n_features):
            var offset = Float64(rng.next_int(2000) - 1000) / 1000.0
            X[i, j] = centroids[c, j] + offset

    return (X^, y^)


def make_synthetic_blobs(
    n_samples: Int, n_features: Int, n_clusters: Int = 5, seed: Int = 42
) -> Matrix[DType.float64]:
    """Generates synthetic clustered blobs for clustering algorithms."""
    var rng = PRNG(seed)
    var X = Matrix[DType.float64](n_samples, n_features, 0.0)

    var centers = Matrix[DType.float64](n_clusters, n_features, 0.0)
    for k in range(n_clusters):
        for j in range(n_features):
            centers[k, j] = (
                Float64(k * 5) + Float64(rng.next_int(200) - 100) / 50.0
            )

    for i in range(n_samples):
        var k = i % n_clusters
        for j in range(n_features):
            var noise = Float64(rng.next_int(2000) - 1000) / 1000.0
            X[i, j] = centers[k, j] + noise

    return X^


def make_synthetic_sparse(
    rows: Int, cols: Int, nnz_per_row: Int = 10, seed: Int = 42
) raises -> CSRMatrix[DType.float64]:
    """Generates synthetic CSR sparse matrix."""
    var rng = PRNG(seed)
    var total_nnz = rows * nnz_per_row
    var data = List[Scalar[DType.float64]](capacity=total_nnz)
    var indices = List[Int](capacity=total_nnz)
    var indptr = List[Int](capacity=rows + 1)
    indptr.append(0)

    for _ in range(rows):
        for _ in range(nnz_per_row):
            var c = rng.next_int(cols)
            var val = Float64(rng.next_int(1000) + 1) / 100.0
            data.append(val)
            indices.append(c)
        indptr.append(len(data))

    return CSRMatrix[DType.float64](rows, cols, data^, indices^, indptr^)


def make_synthetic_counts(
    n_samples: Int, n_features: Int, n_classes: Int = 2, seed: Int = 42
) -> Tuple[Matrix[DType.float64], List[Scalar[DType.int32]]]:
    """Generates synthetic discrete count dataset for Naive Bayes benchmarks."""
    var rng = PRNG(seed)
    var X = Matrix[DType.float64](n_samples, n_features, 0.0)
    var y = List[Scalar[DType.int32]](capacity=n_samples)

    for i in range(n_samples):
        var c = i % n_classes
        y.append(Int32(c))
        for j in range(n_features):
            var base_rate = 5 if (j % n_classes) == c else 1
            var count_val = Float64(rng.next_int(base_rate * 3 + 1))
            X[i, j] = count_val

    return (X^, y^)
