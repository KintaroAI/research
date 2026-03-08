"""
Choose 2 random points within some area and calculate cost function
for all dots around certain radius from chose points

"""

import cv2
import numpy as np
import random
import math
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

def create_weight_matrix(neurons_count):
    weight_matrix = np.zeros((neurons_count, neurons_count))
    for i in range(neurons_count):
        for j in range(i, neurons_count):
            if i == j:
                weight = 2.0
            else:
                weight = 1/abs(i-j)
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight
    return weight_matrix

def cost_function(matrix, weight_matrix, p1, p2, pmin, pmax):
    cost = 0
    radius = 3
    for i in range(max(p1.x-radius, pmin.x), min(p1.x+radius+1, pmax.x+1)):
        for j in range(max(p1.y-radius, pmin.y), min(p1.y+radius+1, pmax.y+1)):
            for k in range(max(p2.x-radius, pmin.x), min(p2.x+radius+1, pmax.x+1)):
                for l in range(max(p2.y-radius, pmin.y), min(p2.y+radius+1, pmax.y+1)):
                    dist = abs(i - k) + abs(j - l)
                    weight = weight_matrix[matrix[i, j], matrix[k, l]]
                    cost += weight * dist
    return cost

def simulated_annealing(matrix, weight_matrix, init_temp=100, cooling_rate=0.99, iterations=10000):
    temp = init_temp
    p_border_min = Point(0, 0)
    p_border_max = Point(matrix.shape[0] - 1, matrix.shape[1] - 1)
    radius = round(0.1*min(p_border_max)) or 3
    p = Point(random.randint(p_border_min.x, p_border_max.x), random.randint(p_border_min.y, p_border_max.y))
    pmin = Point(
        max(p_border_min.x, p.x-radius),
        max(p_border_min.y, p.y-radius)
    )
    pmax = Point(
        min(p_border_max.x, p.x+radius),
        min(p_border_max.y, p.y+radius)
    )

    p1 = Point(random.randint(pmin.x, pmax.x), random.randint(pmin.y, pmax.y))
    for _ in range(iterations):
        p1 = Point(random.randint(pmin.x, pmax.x), random.randint(pmin.y, pmax.y))
        p2 = Point(random.randint(pmin.x, pmax.x), random.randint(pmin.y, pmax.y))
        if p1 == p2:
            continue
        current_cost = cost_function(matrix, weight_matrix, p1, p2, pmin, pmax)
        matrix[p1], matrix[p2] = matrix[p2], matrix[p1]
        new_cost = cost_function(matrix, weight_matrix, p1, p2, pmin, pmax)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or math.exp(-delta_cost / temp) > random.random():
            current_cost = new_cost
        else:
            matrix[p1], matrix[p2] = matrix[p2], matrix[p1]

        temp *= cooling_rate

    return temp

# Example usage
rows, columns = 48, 64
#rows, columns = 10, 10
neurons_count = rows * columns

weight_matrix = create_weight_matrix(neurons_count)
initial_matrix = np.random.permutation(neurons_count).reshape((rows, columns))
print("Initial matrix:")
#print(initial_matrix)

matrix = initial_matrix

temp = 100
while True:
    normalized_frame = cv2.normalize(
        matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Normalized Frame", normalized_frame)
    #normalized_weights = cv2.normalize(
    #    weight_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imshow("Weights", normalized_weights)
    cv2.waitKey(1)
    temp = simulated_annealing(initial_matrix, weight_matrix, iterations=1000, cooling_rate=0.999, init_temp=temp)

print("\nOptimized matrix:")
print(matrix)
cv2.destroyAllWindows()

