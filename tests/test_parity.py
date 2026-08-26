"""Behavioural parity checks against the installed pynndescent package."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

upstream = pytest.importorskip("pynndescent")

from mojo_pynndescent import NNDescent, PyNNDescentTransformer


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(72)
    return np.ascontiguousarray(rng.normal(size=(180, 7)))


def _recall(left, right):
    return np.mean([len(set(a).intersection(b)) / len(a) for a, b in zip(left, right)])


def _brute(data, query, k, metric):
    if metric == "euclidean":
        dist = np.sqrt(((query[:, None] - data[None, :]) ** 2).sum(axis=2))
    elif metric == "sqeuclidean":
        dist = ((query[:, None] - data[None, :]) ** 2).sum(axis=2)
    elif metric == "cosine":
        q = query / np.linalg.norm(query, axis=1, keepdims=True)
        x = data / np.linalg.norm(data, axis=1, keepdims=True)
        dist = 1.0 - q @ x.T
    else:
        dist = np.abs(query[:, None] - data[None, :]).sum(axis=2)
    idx = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(dist, idx, axis=1)


def test_constructor_signature_matches_upstream():
    assert list(inspect.signature(NNDescent).parameters) == list(inspect.signature(upstream.NNDescent).parameters)


def test_neighbor_graph_agrees_with_upstream(data):
    ours = NNDescent(data, n_neighbors=12, n_iters=8, random_state=42)
    theirs = upstream.NNDescent(data, n_neighbors=12, n_iters=8, random_state=42, n_jobs=1)
    our_idx, our_dist = ours.neighbor_graph
    their_idx, their_dist = theirs.neighbor_graph
    assert np.array_equal(our_idx[:, 0], np.arange(len(data)))
    assert np.all(np.diff(our_dist, axis=1) >= 0)
    assert _recall(our_idx, their_idx) >= 0.90
    shared_distance_error = []
    for row in range(len(data)):
        where = {node: col for col, node in enumerate(their_idx[row])}
        for col, node in enumerate(our_idx[row]):
            if node in where:
                shared_distance_error.append(abs(our_dist[row, col] - their_dist[row, where[node]]))
    # Upstream's numba graph uses float32 intermediates even for float64 input.
    assert max(shared_distance_error) < 5e-6


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "cosine", "manhattan", "cityblock"])
def test_query_matches_exact_reference(data, metric):
    query = np.ascontiguousarray(data[5:30] + 0.031)
    ours = NNDescent(data, metric=metric, n_neighbors=12, n_iters=8, random_state=3)
    our_idx, our_dist = ours.query(query, k=8)
    exact_idx, exact_dist = _brute(data, query, 8, metric)
    assert np.array_equal(our_idx, exact_idx)
    assert np.allclose(our_dist, exact_dist, atol=1e-10)


def test_large_tail_dimension_query_matches_exact_reference():
    rng = np.random.default_rng(19)
    data = np.ascontiguousarray(rng.normal(size=(600, 7)))
    query = np.ascontiguousarray(data[:256] + 0.017)
    index = NNDescent(data, n_neighbors=8, n_iters=0, random_state=5)
    our_idx, our_dist = index.query(query, k=8)
    exact_idx, exact_dist = _brute(data, query, 8, "euclidean")
    assert np.array_equal(our_idx, exact_idx)
    assert np.allclose(our_dist, exact_dist, atol=1e-10)


def test_large_graph_initialization_parallel_threshold():
    rng = np.random.default_rng(23)
    data = np.ascontiguousarray(rng.normal(size=(2048, 64)))
    index = NNDescent(data, n_neighbors=8, n_iters=0, random_state=7)
    indices, distances = index.neighbor_graph
    assert np.array_equal(indices[:, 0], np.arange(len(data)))
    assert np.all(np.diff(distances, axis=1) >= 0)
    assert np.allclose(distances[:, 0], 0.0)


def test_vectorized_initial_graph_has_unique_valid_neighbors():
    rng = np.random.default_rng(29)
    data = np.ascontiguousarray(rng.normal(size=(257, 9)))
    index = NNDescent(data, n_neighbors=31, n_iters=0, random_state=11)
    indices, _ = index.neighbor_graph
    assert np.all((indices >= 0) & (indices < len(data)))
    assert all(len(set(row)) == len(row) for row in indices)


def test_dense_initial_graph_has_unique_valid_neighbors():
    data = np.arange(40.0).reshape(20, 2)
    index = NNDescent(data, n_neighbors=18, n_iters=0, random_state=13)
    indices, _ = index.neighbor_graph
    assert np.all((indices >= 0) & (indices < len(data)))
    assert all(len(set(row)) == len(row) for row in indices)


def test_zero_round_euclidean_graph_has_public_distances():
    data = np.ascontiguousarray([[0.0], [3.0], [7.0]])
    index = NNDescent(data, n_neighbors=1, n_iters=0, random_state=1)
    assert np.array_equal(index.neighbor_graph[0].ravel(), np.arange(len(data)))
    assert np.array_equal(index.neighbor_graph[1].ravel(), np.zeros(len(data)))


def test_update_rebuilds_and_preserves_query_contract(data):
    index = NNDescent(data, n_neighbors=10, random_state=8, n_iters=7)
    updated = data[[2, 9]] + 4.0
    index.update(xs_updated=updated, updated_indices=np.array([2, 9]))
    expected_idx, expected_dist = _brute(index.data, index.data[[2, 9]], 4, "euclidean")
    idx, dist = index.query(index.data[[2, 9]], k=4)
    assert np.array_equal(idx, expected_idx)
    assert np.allclose(dist, expected_dist)
    index.update(xs_fresh=np.ones((2, data.shape[1])))
    assert index.neighbor_graph[0].shape == (len(data) + 2, 10)


def test_prepare_and_compress_index_are_supported(data):
    index = NNDescent(data, n_neighbors=6, random_state=2)
    assert index.prepare() is index
    assert index.compress_index() is index
    assert index.compressed is True


def test_transformer_returns_sparse_knn_graph(data):
    transformer = PyNNDescentTransformer(n_neighbors=6, n_iters=7, random_state=4)
    graph = transformer.fit_transform(data)
    assert graph.shape == (len(data), len(data))
    assert graph.nnz == len(data) * 6
    assert np.allclose(graph.diagonal(), 0.0)
    fresh = transformer.transform(data[:5])
    assert fresh.shape == (5, len(data))
    assert fresh.nnz == 5 * 6


def test_input_validation(data):
    with pytest.raises(ValueError, match="unsupported metric"):
        NNDescent(data, metric="hamming")
    with pytest.raises(ValueError, match="n_neighbors"):
        NNDescent(data, n_neighbors=len(data))
    with pytest.raises(ValueError, match="n_neighbors must be an integer"):
        NNDescent(data, n_neighbors=2.5)
    with pytest.raises(ValueError, match="integer dtype"):
        NNDescent(data, n_neighbors=6, init_graph=np.zeros((len(data), 6)))
    with pytest.raises(ValueError, match="exactly representable"):
        NNDescent(np.array([[0, 1], [2**54, 3]], dtype=np.int64), n_neighbors=1)
    index = NNDescent(data, n_neighbors=6, random_state=0)
    assert index.query(data[:1], k=2)[0].shape == (1, 2)
    with pytest.raises(ValueError, match="features"):
        index.query(np.ones((3, 2)))
