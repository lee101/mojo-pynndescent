"""A compact, NumPy-compatible port of pynndescent's NNDescent index."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from ._lib import lib

_METRICS = {"euclidean": 0, "sqeuclidean": 1, "cosine": 2, "manhattan": 3, "cityblock": 3}


def _array(data: Any, min_rows: int = 2) -> np.ndarray:
    value = np.asarray(data)
    if value.ndim != 2 or value.shape[0] < min_rows or value.shape[1] < 1:
        raise ValueError(f"data must be a two-dimensional array with at least {min_rows} rows")
    if value.dtype.kind not in "iuf" or value.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise ValueError("data must use a real numeric dtype no wider than float64")
    # Integer values beyond this limit cannot be represented exactly by the
    # float64 ABI.  Rejecting them is preferable to silently changing points.
    if value.dtype.kind == "i" and (value.min() < -(2**53) or value.max() > 2**53):
        raise ValueError("integer data must be exactly representable as float64")
    if value.dtype.kind == "u" and value.max() > 2**53:
        raise ValueError("integer data must be exactly representable as float64")
    value = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("data must contain only finite values")
    return value


def _count(value: Any, name: str, lower: int, upper: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        end = str(upper) if upper is not None else "infinity"
        raise ValueError(f"{name} must be in [{lower}, {end}]")
    return value


def _indices(data: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(data)
    if value.shape != shape or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must have shape {shape} and an integer dtype")
    if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains values outside int64")
    return np.ascontiguousarray(value, dtype=np.int64)


def _metric_code(metric: str | Any) -> int:
    if callable(metric):
        raise ValueError("callable metrics are not covered by the Mojo backend")
    try:
        return _METRICS[metric.lower()]
    except (AttributeError, KeyError) as exc:
        supported = ", ".join(sorted(_METRICS))
        raise ValueError(f"unsupported metric {metric!r}; supported metrics: {supported}") from exc


class NNDescent:
    """Approximate nearest-neighbour graph built by iterative NN-descent.

    The constructor and :meth:`query` follow pynndescent. Parameters that tune
    random-projection trees or numba parallelism are accepted for drop-in use;
    this implementation uses random graph seeding and a deterministic serial
    refinement kernel instead.
    """

    def __init__(
        self, data, metric="euclidean", metric_kwds=None, n_neighbors=30,
        n_trees=None, leaf_size=None, pruning_degree_multiplier=1.5,
        diversify_prob=1.0, diversify_method="standard",
        degree_prune_aggressiveness=1.0, n_search_trees=1,
        search_tree_leaf_size=None, max_search_tree_depth=None, quantization=None,
        tree_init=True, init_graph=None, init_dist=None, random_state=None,
        low_memory=True, max_candidates=None, max_rptree_depth=200, n_iters=None,
        delta=0.001, n_jobs=None, compressed=False,
        parallel_batch_queries=False, verbose=False,
    ):
        self._raw_data = _array(data)
        self.data = self._raw_data
        self.metric = metric
        self.metric_kwds = {} if metric_kwds is None else dict(metric_kwds)
        if self.metric_kwds:
            raise ValueError("metric_kwds are not covered by the Mojo backend")
        self.n_neighbors = _count(n_neighbors, "n_neighbors", 1, len(self.data) - 1)
        self.n_trees, self.leaf_size = n_trees, leaf_size
        self.pruning_degree_multiplier, self.diversify_prob = pruning_degree_multiplier, diversify_prob
        self.diversify_method = diversify_method
        self.degree_prune_aggressiveness = degree_prune_aggressiveness
        self.n_search_trees, self.tree_init = n_search_trees, tree_init
        self.search_tree_leaf_size = search_tree_leaf_size
        self.max_search_tree_depth, self.quantization = max_search_tree_depth, quantization
        self.random_state = random_state
        self.low_memory, self.max_candidates = low_memory, max_candidates
        self.max_rptree_depth, self.delta, self.n_jobs = max_rptree_depth, delta, n_jobs
        self.compressed, self.parallel_batch_queries, self.verbose = compressed, parallel_batch_queries, verbose
        self._metric = _metric_code(metric)
        self.n_iters = (
            _count(n_iters, "n_iters", 0)
            if n_iters is not None else max(5, int(np.ceil(np.log2(len(self.data)))))
        )
        self._rng = np.random.default_rng(random_state)
        self._neighbor_graph: tuple[np.ndarray, np.ndarray] | None = None
        self._build(init_graph, init_dist)

    def _initial_graph(self) -> np.ndarray:
        n, k = len(self.data), self.n_neighbors
        result = np.empty((n, k), dtype=np.int64)
        for row in range(n):
            result[row, 0] = row
            if k > 1:
                sample = self._rng.choice(n - 1, size=k - 1, replace=False)
                result[row, 1:] = sample + (sample >= row)
        return result

    def _build(self, init_graph, init_dist) -> None:
        n, d = self.data.shape
        if init_graph is None:
            indices = self._initial_graph()
        else:
            indices = _indices(init_graph, (n, self.n_neighbors), "init_graph")
            if (indices < 0).any() or (indices >= n).any() or any(len(set(row)) != len(row) for row in indices):
                raise ValueError("init_graph must have unique valid indices in every row")
            indices = indices.copy()
        distances = np.empty((n, self.n_neighbors), dtype=np.float64)
        backend = lib()
        backend.mpd_initialize(self.data.ctypes.data, indices.ctypes.data, distances.ctypes.data, n, d, self.n_neighbors, self._metric)
        if init_dist is not None:
            supplied = _array(init_dist, min_rows=n)
            if supplied.shape != distances.shape:
                raise ValueError("init_dist must have shape (n_samples, n_neighbors)")
        self._refinement_updates = backend.mpd_refine(
            self.data.ctypes.data, indices.ctypes.data, distances.ctypes.data,
            n, d, self.n_neighbors, self._metric, self.n_iters,
        )
        self._neighbor_graph = indices, distances
        self._search_graph = indices
        self._distance_correction = None
        self._is_sparse = False
        self._rp_forest = None

    @property
    def neighbor_graph(self):
        """The ``(indices, distances)`` graph, sorted nearest-first per row."""
        return self._neighbor_graph

    def prepare(self):
        """Match pynndescent's eager query-preparation hook."""
        return self

    def query(self, query_data, k=10, epsilon=0.1, proxy_beam_size=4):
        query = _array(query_data, min_rows=1)
        if query.shape[1] != self.data.shape[1]:
            raise ValueError("query_data has a different number of features")
        k = _count(k, "k", 1, len(self.data))
        indices = np.empty((len(query), k), dtype=np.int64)
        distances = np.empty((len(query), k), dtype=np.float64)
        lib().mpd_query(
            self.data.ctypes.data, query.ctypes.data, indices.ctypes.data, distances.ctypes.data,
            len(self.data), self.data.shape[1], len(query), k, self._metric,
        )
        return indices, distances

    def update(self, xs_fresh=None, xs_updated=None, updated_indices=None):
        """Rebuild after applying pynndescent-compatible data updates."""
        if xs_updated is not None:
            if updated_indices is None:
                raise ValueError("updated_indices is required with xs_updated")
            ids_input = np.asarray(updated_indices)
            ids = _indices(updated_indices, ids_input.shape, "updated_indices")
            if ids.ndim != 1:
                raise ValueError("updated_indices must be one-dimensional")
            values = _array(xs_updated, min_rows=1)
            if values.shape != (len(ids), self.data.shape[1]):
                raise ValueError("xs_updated must match updated_indices and feature count")
            if (ids < 0).any() or (ids >= len(self.data)).any():
                raise ValueError("updated_indices contains an invalid row")
            self.data = self.data.copy()
            self.data[ids] = values
        if xs_fresh is not None:
            fresh = _array(xs_fresh, min_rows=1)
            if fresh.shape[1] != self.data.shape[1]:
                raise ValueError("xs_fresh has a different number of features")
            self.data = np.ascontiguousarray(np.vstack((self.data, fresh)))
        self._raw_data = self.data
        self._build(None, None)
        return self

    def compress_index(self):
        self.compressed = True
        return self


def make_nn_descent(distance_func, dist_args, n_neighbors, rng_state, **kwargs):
    """Compatibility factory for the common upstream helper entry point.

    Unlike upstream's numba-specialized function factory this returns a Python
    callable that builds :class:`NNDescent` instances using the selected metric.
    """
    metric = kwargs.pop("metric", "euclidean")
    if distance_func is not None and metric == "euclidean":
        raise ValueError("custom distance_func is not covered by the Mojo backend")

    def build(data, *args, **more):
        return NNDescent(data, metric=metric, n_neighbors=n_neighbors, random_state=rng_state, **kwargs, **more)

    return build


class PyNNDescentTransformer:
    """scikit-learn-style transformer returning a sparse KNN distance graph."""

    def __init__(
        self, n_neighbors=30, metric="euclidean", metric_kwds=None, n_trees=None,
        leaf_size=None, search_epsilon=0.1, pruning_degree_multiplier=1.5,
        diversify_prob=1.0, n_search_trees=1, tree_init=True, random_state=None,
        n_jobs=None, low_memory=True, max_candidates=None, n_iters=None,
        early_termination_value=0.001, parallel_batch_queries=False, verbose=False,
    ):
        self.n_neighbors, self.metric, self.metric_kwds = n_neighbors, metric, metric_kwds
        self.n_trees, self.leaf_size, self.search_epsilon = n_trees, leaf_size, search_epsilon
        self.pruning_degree_multiplier, self.diversify_prob = pruning_degree_multiplier, diversify_prob
        self.n_search_trees, self.tree_init, self.random_state = n_search_trees, tree_init, random_state
        self.n_jobs, self.low_memory, self.max_candidates, self.n_iters = n_jobs, low_memory, max_candidates, n_iters
        self.early_termination_value = early_termination_value
        self.parallel_batch_queries, self.verbose = parallel_batch_queries, verbose
        self.index_: NNDescent | None = None

    def fit(self, X, compress_index=True):
        self.index_ = NNDescent(
            X, metric=self.metric, metric_kwds=self.metric_kwds, n_neighbors=self.n_neighbors,
            n_trees=self.n_trees, leaf_size=self.leaf_size,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
            diversify_prob=self.diversify_prob, n_search_trees=self.n_search_trees,
            tree_init=self.tree_init, random_state=self.random_state, n_jobs=self.n_jobs,
            low_memory=self.low_memory, max_candidates=self.max_candidates,
            n_iters=self.n_iters, delta=self.early_termination_value,
            parallel_batch_queries=self.parallel_batch_queries, verbose=self.verbose,
            compressed=compress_index,
        )
        self._fit_X = self.index_.data
        return self

    def transform(self, X, y=None):
        if self.index_ is None:
            raise ValueError("PyNNDescentTransformer is not fitted")
        from scipy.sparse import csr_matrix

        indices, distances = self.index_.query(X, k=self.n_neighbors, epsilon=self.search_epsilon)
        rows = np.repeat(np.arange(len(indices)), self.n_neighbors)
        return csr_matrix((distances.ravel(), (rows, indices.ravel())), shape=(len(indices), len(self._fit_X)))

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        from scipy.sparse import csr_matrix

        indices, distances = self.index_.neighbor_graph
        rows = np.repeat(np.arange(len(indices)), self.n_neighbors)
        return csr_matrix((distances.ravel(), (rows, indices.ravel())), shape=(len(indices), len(indices)))

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in (
            "n_neighbors", "metric", "metric_kwds", "n_trees", "leaf_size",
            "search_epsilon", "pruning_degree_multiplier", "diversify_prob",
            "n_search_trees", "tree_init", "random_state", "n_jobs", "low_memory",
            "max_candidates", "n_iters", "early_termination_value",
            "parallel_batch_queries", "verbose",
        )}

    def set_params(self, **params):
        for name, value in params.items():
            if name not in self.get_params():
                raise ValueError(f"invalid parameter {name!r}")
            setattr(self, name, value)
        return self
