"""NN-descent kernels exported to the small Python ctypes layer."""

from std.algorithm import parallelize
from std.math import sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime W = simdwidthof[DType.float64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_DISTANCE_WORK = 1048576


def distance(a: FPtr, b: FPtr, d: Int, metric: Int) -> Float64:
    if metric == 2:
        var dots = SIMD[DType.float64, W](0.0)
        var aas = SIMD[DType.float64, W](0.0)
        var bbs = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= d:
            var av = a.load[width=W](j)
            var bv = b.load[width=W](j)
            dots += av * bv
            aas += av * av
            bbs += bv * bv
            j += W
        var dot = dots.reduce_add()
        var aa = aas.reduce_add()
        var bb = bbs.reduce_add()
        while j < d:
            dot += a[j] * b[j]
            aa += a[j] * a[j]
            bb += b[j] * b[j]
            j += 1
        if aa == 0.0 or bb == 0.0:
            return 1.0
        return 1.0 - dot / sqrt(aa * bb)
    if metric == 3:
        var totals = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= d:
            var delta = a.load[width=W](j) - b.load[width=W](j)
            totals += max(delta, -delta)
            j += W
        var total = totals.reduce_add()
        while j < d:
            var delta = a[j] - b[j]
            total += -delta if delta < 0.0 else delta
            j += 1
        return total
    var acc = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + W <= d:
        var delta = a.load[width=W](j) - b.load[width=W](j)
        acc += delta * delta
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = a[j] - b[j]
        total += delta * delta
        j += 1
    return total if metric == 1 else sqrt(total)


def graph_distance(a: FPtr, b: FPtr, d: Int, metric: Int) -> Float64:
    if metric != 0:
        return distance(a, b, d, metric)
    var acc = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + W <= d:
        var delta = a.load[width=W](j) - b.load[width=W](j)
        acc += delta * delta
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = a[j] - b[j]
        total += delta * delta
        j += 1
    return total


def sort_row(indices: IPtr, distances: FPtr, base: Int, k: Int):
    for pos in range(1, k):
        var p = pos
        while p > 0 and distances[base + p - 1] > distances[base + p]:
            var td = distances[base + p]
            distances[base + p] = distances[base + p - 1]
            distances[base + p - 1] = td
            var ti = indices[base + p]
            indices[base + p] = indices[base + p - 1]
            indices[base + p - 1] = ti
            p -= 1


def insert(indices: IPtr, distances: FPtr, base: Int, k: Int, candidate: Int, value: Float64) -> Bool:
    if value >= distances[base + k - 1]:
        return False
    for pos in range(k):
        if indices[base + pos] == Int64(candidate):
            return False
    var pos = k - 1
    while pos > 0 and distances[base + pos - 1] > value:
        distances[base + pos] = distances[base + pos - 1]
        indices[base + pos] = indices[base + pos - 1]
        pos -= 1
    distances[base + pos] = value
    indices[base + pos] = Int64(candidate)
    return True


def initialize_row(
    data: FPtr, indices: IPtr, distances: FPtr,
    row: Int, d: Int, k: Int, metric: Int,
):
    var base = row * k
    for col in range(k):
        var neighbour = Int(indices[base + col])
        distances[base + col] = graph_distance(
            data + row * d, data + neighbour * d, d, metric
        )
    sort_row(indices, distances, base, k)


@export("mpd_initialize")
def mpd_initialize(
    data_addr: Int, indices_addr: Int, distances_addr: Int,
    n: Int, d: Int, k: Int, metric: Int
) abi("C"):
    var data = FPtr(unsafe_from_address=data_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var distances = FPtr(unsafe_from_address=distances_addr)
    if n * k * d >= PARALLEL_DISTANCE_WORK:
        @parameter
        def initialize_one(row: Int):
            initialize_row(data, indices, distances, row, d, k, metric)
        parallelize[initialize_one](n, 18)
    else:
        for row in range(n):
            initialize_row(data, indices, distances, row, d, k, metric)


@export("mpd_refine")
def mpd_refine(
    data_addr: Int, indices_addr: Int, distances_addr: Int,
    n: Int, d: Int, k: Int, metric: Int, rounds: Int
) abi("C") -> Int:
    var data = FPtr(unsafe_from_address=data_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var distances = FPtr(unsafe_from_address=distances_addr)
    var changes = 0
    for _ in range(rounds):
        var round_changes = 0
        for row in range(n):
            var base = row * k
            for edge in range(k):
                var neighbour = Int(indices[base + edge])
                if neighbour < 0 or neighbour >= n:
                    continue
                var neighbour_base = neighbour * k
                for candidate_slot in range(k):
                    var candidate = Int(indices[neighbour_base + candidate_slot])
                    if candidate == row or candidate < 0 or candidate >= n:
                        continue
                    var value = graph_distance(data + row * d, data + candidate * d, d, metric)
                    if insert(indices, distances, base, k, candidate, value):
                        round_changes += 1
                    if insert(indices, distances, candidate * k, k, row, value):
                        round_changes += 1
        changes += round_changes
        if round_changes == 0:
            break
    # Initialisation may be the only graph pass (``rounds == 0``), so the
    # public Euclidean distances must be converted on every exit path.
    if metric == 0:
        for pos in range(n * k):
            distances[pos] = sqrt(distances[pos])
    return changes


def query_row(
    data: FPtr, query: FPtr, indices: IPtr, distances: FPtr,
    row: Int, n: Int, d: Int, k: Int, metric: Int,
):
    var base = row * k
    for col in range(k):
        indices[base + col] = -1
        distances[base + col] = 1.7976931348623157e308
    for candidate in range(n):
        var value = distance(query + row * d, data + candidate * d, d, metric)
        _ = insert(indices, distances, base, k, candidate, value)


@export("mpd_query")
def mpd_query(
    data_addr: Int, query_addr: Int, indices_addr: Int, distances_addr: Int,
    n: Int, d: Int, m: Int, k: Int, metric: Int
) abi("C"):
    var data = FPtr(unsafe_from_address=data_addr)
    var query = FPtr(unsafe_from_address=query_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var distances = FPtr(unsafe_from_address=distances_addr)
    if m * n * d >= PARALLEL_DISTANCE_WORK:
        @parameter
        def query_one(row: Int):
            query_row(data, query, indices, distances, row, n, d, k, metric)
        parallelize[query_one](m, 18)
    else:
        for row in range(m):
            query_row(data, query, indices, distances, row, n, d, k, metric)
