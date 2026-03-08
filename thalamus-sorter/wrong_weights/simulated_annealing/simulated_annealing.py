import numpy as np
import random
import math

def cost_function(matrix, graph):
    cost = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k, weight in graph[matrix[i, j]]:
                x, y = np.where(matrix == k)
                dist = abs(i - x[0]) + abs(j - y[0])
                cost += weight * dist
    return cost

def swap_neurons(matrix):
    x1, y1 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
    x2, y2 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
    matrix[x1, y1], matrix[x2, y2] = matrix[x2, y2], matrix[x1, y1]

def simulated_annealing_slow(matrix, graph, init_temp=100, cooling_rate=0.99, iterations=10000):
    current_cost = cost_function(matrix, graph)
    temp = init_temp

    for _ in range(iterations):
        prev_matrix = matrix.copy()
        swap_neurons(matrix)

        new_cost = cost_function(matrix, graph)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or math.exp(-delta_cost / temp) > random.random():
            current_cost = new_cost
        else:
            matrix = prev_matrix

        temp *= cooling_rate

    return matrix

def simulated_annealing(matrix, graph, init_temp=100, cooling_rate=0.99, iterations=100000):
    current_cost = cost_function(matrix, graph)
    temp = init_temp

    for _ in range(iterations):
        x1, y1 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
        x2, y2 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
        matrix[x1, y1], matrix[x2, y2] = matrix[x2, y2], matrix[x1, y1]

        new_cost = cost_function(matrix, graph)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or math.exp(-delta_cost / temp) > random.random():
            current_cost = new_cost
        else:
            matrix[x1, y1], matrix[x2, y2] = matrix[x2, y2], matrix[x1, y1]

        temp *= cooling_rate

    return matrix

# Example usage
rows, columns = 5, 5
neurons_count = rows * columns
initial_matrix = np.random.permutation(neurons_count).reshape((rows, columns))
#graph = {i: [(j, abs(initial_matrix[i//columns, i%columns] - initial_matrix[j//columns, j%columns]) == 1 and 1 or 0) for j in range(neurons_count) if i != j] for i in range(neurons_count)}
graph = np.zeros(input_dim, output_dim)

print(graph)
print("Initial matrix:")
print(initial_matrix)

optimized_matrix = simulated_annealing(initial_matrix, graph)
print("\nOptimized matrix:")
print(optimized_matrix)
