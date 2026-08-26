# mojo-pynndescent

`mojo-pynndescent` is a standalone Mojo port of the compute-heavy core of
[pynndescent](https://github.com/lmcinnes/pynndescent): building an approximate
k-nearest-neighbour graph with NN-descent. It exposes a NumPy-friendly Python
API through `ctypes`, without requiring a Python extension module.

## Covered subset

`mojo_pynndescent.NNDescent` mirrors the upstream constructor, `neighbor_graph`,
`query`, `prepare`, `update`, and `compress_index` APIs. The graph builder uses
random unique seeding followed by iterative neighbour-of-neighbour refinement
and reciprocal insertion. `PyNNDescentTransformer` provides the corresponding
scikit-learn-style sparse graph transformer.

The covered dense metrics are `euclidean`, `sqeuclidean`, `cosine`,
`manhattan`, and `cityblock`. Query returns exact KNN results for these metrics;
the approximate NN-descent graph remains useful for graph-based workflows and
is the compute-bound Mojo implementation in this repository.

Not covered: sparse matrices, callable metrics and `metric_kwds`, RP-tree
seeding/search, quantization, upstream's search-graph traversal, and upstream's
advanced diversification/pruning controls. Those constructor options are
accepted and retained for source compatibility, but do not alter the current
Mojo algorithm. `init_dist` is validated but distances are recomputed so the
graph cannot contain stale values.

## Install and use

```bash
pixi install
pixi run build
```

```python
import numpy as np
from mojo_pynndescent import NNDescent

points = np.array([[0., 0.], [1., 0.], [0., 2.], [9., 9.]])
index = NNDescent(points, n_neighbors=3, random_state=0)

graph_indices, graph_distances = index.neighbor_graph
indices, distances = index.query([[0.2, 0.1]], k=2)
assert indices.tolist() == [[0, 1]]
assert np.allclose(distances, [[np.sqrt(0.05), np.sqrt(0.65)]])
```

Save the snippet as `examples/basic.py`, then run it from the checkout with
`pixi run python examples/basic.py`; Pixi sets
`PYTHONPATH=python` for each task.

## Correctness

`tests/test_parity.py` installs and exercises real `pynndescent 0.6.0`. On a
fixed dense dataset it asserts graph self-neighbours, sorted graph distances,
at least 90% neighbour-set overlap with upstream, and compatible graph
distances (allowing upstream's float32 numba intermediates). It also verifies
all covered query metrics against an exact NumPy reference, updates, sparse
transformer output, signature parity, and input validation.

The upstream package's `prepare()`/`query()` path aborts in the pinned Python
3.13 environment, so this repository does not claim a direct runtime-query
benchmark against it. Its graph-construction path is stable and is used for
the upstream parity and benchmark comparison.

## Benchmark

Measured with `pixi run bench` on Linux 6.8.0-136-generic x86_64 (glibc 2.39),
x86_64 CPU. Best of three; upstream is constrained to `n_jobs=1`.

| case | mojo-pynndescent | pynndescent | ratio |
| --- | ---: | ---: | ---: |
| NNDescent build (1200 x 16, k=16) | 63.8 ms | 94.5 ms | 1.48x faster |

Euclidean graph refinement compares squared distances and takes square roots
only once when materializing the public graph. Distance kernels use independent
SIMD accumulators with scalar tails, and random graph seeding batches rows to
avoid per-row NumPy allocations. Exact queries and graph initialization
parallelize independent rows only above 1,048,576 distance-elements; smaller
inputs remain serial to avoid worker-launch overhead.

GPU execution is intentionally not provided: the distance scans have low
arithmetic intensity (roughly three floating-point operations per 16 bytes of
point data) and host/device transfer would lose to the CPU path. The MAX
dependency supplies CPU `parallelize`; it is not used for GPU execution.

## How it works

```
NumPy float64 arrays
        | ctypes: base addresses as signed 64-bit integers
src/capi.mojo
        | rebuilds mutable raw pointers inside C-ABI exports
NN-descent init/refinement and exact query kernels
```

All matrices are contiguous row-major `float64`; graph indices are contiguous
row-major `int64` and graph distances are `float64`. Python allocates and owns
every buffer. Mojo receives only integer addresses, rebuilds
`UnsafePointer[..., AnyOrigin[mut=True]]` inside each exported ABI wrapper, and
does no allocation, so there are no cross-language ownership or release rules.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

## License

MIT
