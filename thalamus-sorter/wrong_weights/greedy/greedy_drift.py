# Swapping correlated neurons
import cv2
import numpy as np
from collections import defaultdict
import random

def next_move(x1, y1, x2, y2):
    # Calculate differences
    x_diff = x2 - x1
    y_diff = y2 - y1

    # Determine the direction for x
    if x_diff > 0:
        x1 += 1
    elif x_diff < 0:
        x1 -= 1

    # Determine the direction for y
    # Only move in y direction if x is already aligned or if the difference in y is greater
    if y_diff > 0:
        y1 += 1
    elif y_diff < 0:
        y1 -= 1

    return x1, y1

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

def greedy_swap(neurons_matrix, weight_matrix):
    h, w = neurons_matrix.shape
    total_neurons = h * w
    for y in range(h):
        for x in range(w):
            maybe_swap(neurons_matrix, weight_matrix, x, y)


def greedy_drift(neurons_matrix, weight_matrix):
    window = 4
    h, w = neurons_matrix.shape
    total_neurons = h * w
    for _ in range(int(0.1*total_neurons)):
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        if random.random() < 0.1:
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
        else:
            x2, y2 = random.randint(x1-window, x1+window), random.randint(y1-window, y1+window)
            if x2 < 0:
                x2 = 0
            if x2 >= w:
                x2 = w-1
            if y2 < 0:
                y2 = 0
            if y2 >= h:
                y2 = h-1
        i = neurons_matrix[y1][x1]
        j = neurons_matrix[y2][x2]
        if weight_matrix[i][j] >= 0.9:
            x1i, y1i = next_move(x1, y1, x2, y2)
            x2i, y2i = next_move(x2, y2, x1, y1)
            neurons_matrix[y1][x1], neurons_matrix[y1i][x1i] = neurons_matrix[y1i][x1i], neurons_matrix[y1][x1]
            neurons_matrix[y2][x2], neurons_matrix[y2i][x2i] = neurons_matrix[y2i][x2i], neurons_matrix[y2][x2]



def maybe_swap(neurons_matrix, weight_matrix, x, y):
    h, w = neurons_matrix.shape
    window = 5
    i = neurons_matrix[y][x]
    for dy in range(y-window, y+window+1):
        if dy < 0 or dy >= h:
            continue
        for dx in range(x-window, x+window+1):
            if dx < 0 or dx >= w:
                continue
            if dx == x and dy == y:
                continue
            j = neurons_matrix[dy][dx]

            if weight_matrix[i][j] >= 0.4: # random.random()*100:
                neurons_matrix[y][x], neurons_matrix[dy][dx] = neurons_matrix[dy][dx], neurons_matrix[y][x]

# Example usage
rows, columns = 20, 20
neurons_count = rows * columns
# For weight matrix [i, j] contains j index of the neuron.
neurons_matrix = np.random.permutation(neurons_count).reshape((rows, columns))
print(neurons_matrix)
weight_matrix = create_weight_matrix(neurons_count)
print(weight_matrix)

for _ in range(1000000):
    greedy_drift(neurons_matrix, weight_matrix)
print(neurons_matrix)

while True:
    greedy_drift(neurons_matrix, weight_matrix)
    #print(neurons_matrix)
    normalized_frame = cv2.normalize(
        neurons_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Normalized Frame", normalized_frame)
    normalized_sorted_neurons_matrix = cv2.normalize(
        neurons_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Sorted", normalized_sorted_neurons_matrix)
    cv2.waitKey(1)

