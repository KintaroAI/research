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

def create_weight_matrix(neurons_count, decay_rate = 0.1):
    weight_matrix = np.zeros((neurons_count, neurons_count))
    for i in range(neurons_count):
        for j in range(i, neurons_count):
            if i == j:
                weight = 1.0
            else:
                distance = abs(i - j)
                weight = max(1 - decay_rate * distance, 0)  # Ensure weight is non-negative
                #weight = 1/(abs(i-j))
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight
    return weight_matrix



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
        if weight_matrix[i][j] >= random.random():
            x1i, y1i = next_move(x1, y1, x2, y2)
            x2i, y2i = next_move(x2, y2, x1, y1)
            neurons_matrix[y1][x1], neurons_matrix[y1i][x1i] = neurons_matrix[y1i][x1i], neurons_matrix[y1][x1]
            neurons_matrix[y2][x2], neurons_matrix[y2i][x2i] = neurons_matrix[y2i][x2i], neurons_matrix[y2][x2]



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

