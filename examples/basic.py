"""Minimal mojo-pynndescent usage example."""

import numpy as np

from mojo_pynndescent import NNDescent


points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [9.0, 9.0]])
index = NNDescent(points, n_neighbors=3, random_state=0)

graph_indices, graph_distances = index.neighbor_graph
indices, distances = index.query([[0.2, 0.1]], k=2)
assert graph_indices.shape == graph_distances.shape == (4, 3)
assert indices.tolist() == [[0, 1]]
assert np.allclose(distances, [[np.sqrt(0.05), np.sqrt(0.65)]])
