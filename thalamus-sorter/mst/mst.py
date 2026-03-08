# Kruskal's Minimum Spanning Tree (MST) algorithm
import cv2
import numpy as np
from collections import defaultdict
import random

def create_weight_matrix(neurons_count):
    weight_matrix = np.zeros((neurons_count, neurons_count))
    for i in range(neurons_count):
        for j in range(i, neurons_count):
            if i == j:
                weight = 1.0
            else:
                weight = 1/abs(i-j)
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight
    return weight_matrix


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1
        return True

def build_edges(neurons_matrix, weight_matrix):
    h, w = neurons_matrix.shape
    edges = []

    for i in range(h * w):
        for j in range(i + 1, h * w):
            weight = weight_matrix[i][j]
            edges.append((i, j, weight))

    random.shuffle(edges)
    return edges

def dfs_iterative(node, graph):
    result = []
    visited = set()
    stack = [node]

    while stack:
        current = stack.pop()

        if current not in visited:
            visited.add(current)
            result.append(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return result


def dfs(node, graph):
    result = []
    visited = set()
    visited.add(node)
    result.append(node)

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, graph, visited, result)


def greedy_sort(neurons_matrix, weight_matrix):
    h, w = neurons_matrix.shape
    total_neurons = h * w
    uf = UnionFind(total_neurons)

    edges = build_edges(neurons_matrix, weight_matrix)
    edges.sort(key=lambda x: x[2], reverse=True)

    spanning_tree_edges = []

    for edge in edges:
        i, j, weight = edge
        if uf.union(i, j):
            spanning_tree_edges.append(edge)

    graph = defaultdict(list)
    for i, j, _ in spanning_tree_edges:
        graph[i].append(j)
        graph[j].append(i)

    sorted_neurons = dfs_iterative(3, graph)

    sorted_neurons_matrix = np.array(sorted_neurons).reshape(h, w)
    return sorted_neurons_matrix

# Example usage
rows, columns = 10, 10
neurons_count = rows * columns
neurons_matrix = np.random.permutation(neurons_count).reshape((rows, columns))
print(neurons_matrix)
weight_matrix = create_weight_matrix(neurons_count)

sorted_neurons_matrix = greedy_sort(neurons_matrix, weight_matrix)
print(sorted_neurons_matrix)

while True:
    normalized_frame = cv2.normalize(
        neurons_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Normalized Frame", normalized_frame)
    normalized_sorted_neurons_matrix = cv2.normalize(
        sorted_neurons_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Sorted", normalized_sorted_neurons_matrix)
    cv2.waitKey(1)

