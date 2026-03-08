import cv2
import cupy as cp
import random

class Sorter:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.neurons_count = self.width * self.height

        # Use CuPy for GPU-accelerated operations
        self.neurons_matrix = cp.random.permutation(self.neurons_count).reshape((self.height, self.width))
        self.weight_matrix = self.create_weight_matrix()
        self.coordinates = self.init_coordinates()
        self.k = 24
        self.image = self.load_image('./cube.png')
        self.output = self.init_output()

    def init_output(self):
        output = cp.zeros(self.neurons_count, dtype=cp.uint8).reshape((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                i = int(self.neurons_matrix[y][x])  # Ensure i is int for indexing
                ox = i % self.width
                oy = i // self.width
                output[y][x] = self.image[oy][ox]
        return output

    def load_image(self, file_name):
        image = cv2.imread(file_name)
        resized_image = cv2.resize(image, (self.height, self.width))

        if resized_image.shape[2] == 4:
            gray_image = cv2.cvtColor(resized_image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        return cp.asarray(gray_image)  # Convert to CuPy array

    def init_coordinates(self):
        coordinates = cp.zeros((self.neurons_count, 2))
        for y in range(self.height):
            for x in range(self.width):
                i = int(self.neurons_matrix[y][x])  # Ensure i is int for indexing
                coordinates[i][0] = x
                coordinates[i][1] = y
        return coordinates

    def create_weight_matrix(self, decay_rate=0.1):
        def get_original_coordinate(i):
            return (i % self.width, i // self.width)

        weight_matrix = cp.zeros((self.neurons_count, self.neurons_count))
        for i in range(self.neurons_count):
            (xi, yi) = get_original_coordinate(i)
            for j in range(i, self.neurons_count):
                if i == j:
                    weight = 1.0
                else:
                    (xj, yj) = get_original_coordinate(j)
                    distance = cp.sqrt((xi-xj)**2 + (yi-yj)**2)
                    weight = max(0, 1-abs(distance)*decay_rate)
                weight_matrix[i, j] = weight
                weight_matrix[j, i] = weight
        return weight_matrix

    def tick(self):
        h, w = self.neurons_matrix.shape
        total_neurons = h * w
        for _ in range(int(0.3*total_neurons)):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            self.greedy_neighbors_move(x, y)

    def get_neighbors(self, x, y):
        i = int(self.neurons_matrix[y][x])  # Ensure i is int for indexing
        kj = cp.argsort(self.weight_matrix[i])[-self.k-1:-1]
        return [(int(self.coordinates[j][0]), int(self.coordinates[j][1])) for j in kj]

    def greedy_neighbors_move(self, x, y):
        neighbors = self.get_neighbors(x, y)

        avg_x = sum(neighbor[0] for neighbor in neighbors) / len(neighbors)
        avg_y = sum(neighbor[1] for neighbor in neighbors) / len(neighbors)

        direction_x = avg_x - x
        direction_y = avg_y - y
        if not direction_x and not direction_y:
            return x, y

        magnitude = cp.sqrt(direction_x**2 + direction_y**2)
        direction_x /= magnitude
        direction_y /= magnitude

        new_x = x + round(direction_x)
        new_y = y + round(direction_y)

        self.swap(x, y, new_x, new_y)
        return new_x, new_y

    def swap(self, x, y, new_x, new_y):
        i = int(self.neurons_matrix[y][x])  # Ensure i is int for indexing
        j = int(self.neurons_matrix[new_y][new_x])  # Ensure j is int for indexing
        self.neurons_matrix[y][x], self.neurons_matrix[new_y][new_x] = self.neurons_matrix[new_y][new_x], self.neurons_matrix[y][x]
        self.coordinates[i][0], self.coordinates[j][0] = self.coordinates[j][0], self.coordinates[i][0]
        self.coordinates[i][1], self.coordinates[j][1] = self.coordinates[j][1], self.coordinates[i][1]
        self.output[y][x], self.output[new_y][new_x] = self.output[new_y][new_x], self.output[y][x]

    def show(self):
        # ... (rest of the show function remains the same)
        pass

# Create Sorter instance and perform operations
sorter = Sorter(80, 80)
for _ in range(100):
    sorter.tick()

# Remember, display operations still need to be done on the CPU

