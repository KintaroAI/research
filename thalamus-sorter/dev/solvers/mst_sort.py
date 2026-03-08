"""MST-based sorting: build maximum spanning tree on weight matrix,
then DFS traversal gives a linear ordering that respects correlation structure."""

import numpy as np
from utils.graph import build_mst, tree_to_adjacency, dfs_order


def mst_sort(weight_matrix, width, height, start_node=0):
    """Sort neurons by building a maximum spanning tree and traversing it.

    Returns a (height, width) matrix where position encodes the DFS ordering,
    placing strongly-connected neurons adjacent in the traversal.
    """
    tree_edges = build_mst(weight_matrix, maximize=True)
    graph = tree_to_adjacency(tree_edges)
    ordering = dfs_order(start_node, graph)
    return np.array(ordering).reshape(height, width)
