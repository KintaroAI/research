"""
Brute-force cost_function against all matrix.
Choose any two points on the matrix and do the swap.
"""


import numpy as np
import random
import math

def create_weight_matrix(neurons_count):
    weight_matrix = np.zeros((neurons_count, neurons_count))
    for i in range(neurons_count):
        for j in range(i+1, neurons_count):
            weight = 1/abs(i-j)
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight
    return weight_matrix

def cost_function(matrix, weight_matrix):
    cost = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[0]):
                for l in range(matrix.shape[1]):
                    dist = abs(i - k) + abs(j - l)
                    weight = weight_matrix[matrix[i, j], matrix[k, l]]
                    cost += weight * dist
    return cost

def simulated_annealing(matrix, weight_matrix, init_temp=100, cooling_rate=0.99, iterations=10000):
    current_cost = cost_function(matrix, weight_matrix)
    temp = init_temp

    for _ in range(iterations):
        x1, y1 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
        x2, y2 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
        matrix[x1, y1], matrix[x2, y2] = matrix[x2, y2], matrix[x1, y1]

        new_cost = cost_function(matrix, weight_matrix)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or math.exp(-delta_cost / temp) > random.random():
            current_cost = new_cost
        else:
            matrix[x1, y1], matrix[x2, y2] = matrix[x2, y2], matrix[x1, y1]

        temp *= cooling_rate

    return matrix

# Example usage
rows, columns = 3, 7
neurons_count = rows * columns

weight_matrix = create_weight_matrix(neurons_count)
initial_matrix = np.random.permutation(neurons_count).reshape((rows, columns))
print("Initial matrix:")
print(initial_matrix)

optimized_matrix = simulated_annealing(initial_matrix, weight_matrix)
print("\nOptimized matrix:")
print(optimized_matrix)

