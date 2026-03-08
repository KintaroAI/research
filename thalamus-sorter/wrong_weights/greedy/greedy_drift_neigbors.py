# Swapping correlated neurons
import cv2
import numpy as np
from collections import defaultdict
import random

class Sorter:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.neurons_count = self.width * self.height
        # Value represent i index in self.weight_matrix
        self.neurons_matrix = np.random.permutation(self.neurons_count).reshape((self.height, self.width))
        self.weight_matrix = self.create_weight_matrix()
        self.coordinates = self.init_coordinates()
        self.k = 8  # KNN count

    def init_coordinates(self):
        coordinates = np.zeros((self.neurons_count, 2))
        for y in range(self.height):
            for x in range(self.width):
                i = self.neurons_matrix[y][x]
                coordinates[i][0] = x
                coordinates[i][1] = y
        return coordinates

    def create_weight_matrix(self, decay_rate = 0.1):
        def get_original_coordinate(i):
            return (i%self.width, i//self.width)

        weight_matrix = np.zeros((self.neurons_count, self.neurons_count))
        for i in range(self.neurons_count):
            (xi, yi) = get_original_coordinate(i)
            for j in range(i, self.neurons_count):
                if i == j:
                    weight = 1.0
                else:
                    (xj, yj) = get_original_coordinate(j)
                    distance = ((xi-xj)**2 + (yi-yj)**2)**0.5
                    weight = 1.0/distance
                    #weight = 1/(abs(i-j))
                weight_matrix[i, j] = weight
                weight_matrix[j, i] = weight
        return weight_matrix

    def get_distance_deprecated(i, j):
        weight_matrix = np.zeros((self.neurons_count, self.neurons_count))
        for i in range(self.neurons_count):
            for j in range(i, self.neurons_count):
                if i == j:
                    weight = 1.0
                else:
                    distance = abs(i - j)
                    weight = max(1 - decay_rate * distance, 0)  # Ensure weight is non-negative
                    #weight = 1/(abs(i-j))
                weight_matrix[i, j] = weight
                weight_matrix[j, i] = weight
        return weight_matrix

    def tick(self):
        h, w = self.neurons_matrix.shape
        total_neurons = h * w
        #for i in range(total_neurons):
        #    x, y = int(self.coordinates[i][0]), int(self.coordinates[i][1])
        #    self.greedy_neighbors_move(x, y)
        for _ in range(int(0.1*total_neurons)):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            self.greedy_neighbors_move(x, y)

    def get_neighbors(self, x, y):
        res = []
        i = self.neurons_matrix[y][x]
        kj = np.argpartition(self.weight_matrix[i], -self.k-1)

        for j in kj[-self.k-1:-1]:
            res.append((self.coordinates[j][0], self.coordinates[j][1]))
        return res

    def greedy_neighbors_move(self, x, y):
        neighbors = self.get_neighbors(x, y)

        # Calculate average position of neighbors
        avg_x = sum(neighbor[0] for neighbor in neighbors) / len(neighbors)
        avg_y = sum(neighbor[1] for neighbor in neighbors) / len(neighbors)

        # Determine direction towards average position
        direction_x = avg_x - x
        direction_y = avg_y - y

        # Normalize the direction to move only one cell
        magnitude = (direction_x**2 + direction_y**2)**0.5
        if magnitude != 0:
            direction_x /= magnitude
            direction_y /= magnitude

        # Move one cell towards the average position
        new_x = x + round(direction_x)
        new_y = y + round(direction_y)

        # Perform the move
        self.swap(x, y, new_x, new_y)

        return new_x, new_y

    def swap(self, x, y, new_x, new_y):
        i = self.neurons_matrix[y][x]
        j = self.neurons_matrix[new_y][new_x]
        self.neurons_matrix[y][x], self.neurons_matrix[new_y][new_x] = self.neurons_matrix[new_y][new_x], self.neurons_matrix[y][x]
        self.coordinates[i][0], self.coordinates[j][0] = self.coordinates[j][0], self.coordinates[i][0]
        self.coordinates[i][1], self.coordinates[j][1] = self.coordinates[j][1], self.coordinates[i][1]

    def show(self):
        weight_matrix_normalized_frame = cv2.normalize(
            self.weight_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Weight matrix", weight_matrix_normalized_frame)

        normalized_frame = cv2.normalize(
            self.neurons_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Sorted", normalized_frame)

        cv2.waitKey(1)


sorter = Sorter(40, 40)

print(sorter.neurons_matrix)
#print(sorter.coordinates)
print(sorter.weight_matrix)

for _ in range(100):
    sorter.tick()
#print(sorter.neurons_matrix)
#print(sorter.coordinates)

while True:
    sorter.tick()
    sorter.show()
