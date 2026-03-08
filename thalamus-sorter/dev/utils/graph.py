"""Graph utilities: UnionFind, edge extraction, traversal."""

import random
from collections import defaultdict


class UnionFind:
    """Disjoint set with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[rx] = ry
            if self.rank[rx] == self.rank[ry]:
                self.rank[ry] += 1
        return True


def build_edges(weight_matrix):
    """Extract all unique edges (i, j, weight) from a symmetric weight matrix.
    Shuffles to break ties randomly in downstream sorting."""
    n = weight_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, weight_matrix[i, j]))
    random.shuffle(edges)
    return edges


def build_mst(weight_matrix, maximize=True):
    """Build a spanning tree using Kruskal's algorithm.
    maximize=True builds maximum spanning tree (highest-weight edges first)."""
    n = weight_matrix.shape[0]
    edges = build_edges(weight_matrix)
    edges.sort(key=lambda e: e[2], reverse=maximize)

    uf = UnionFind(n)
    tree_edges = []
    for i, j, w in edges:
        if uf.union(i, j):
            tree_edges.append((i, j, w))
    return tree_edges


def tree_to_adjacency(tree_edges):
    """Convert edge list to adjacency list."""
    graph = defaultdict(list)
    for i, j, _ in tree_edges:
        graph[i].append(j)
        graph[j].append(i)
    return graph


def dfs_order(start, graph):
    """Iterative DFS traversal returning visit order."""
    result = []
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    return result
