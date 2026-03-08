import cv2
import math
import numpy as np
import random

CAMERA_DIM = (640//2, 480//2)

def get_new_coordinates(ix, iy, jx, jy):
    """Gets new set of coordinates closer to each other"""
    # Compute the Euclidean distance between the original points
    d = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2)
    if d == 0:
        return ix, iy, jx, jy

    # Compute the midpoint of the line segment connecting the original points
    mx = (ix + jx) / 2
    my = (iy + jy) / 2

    # Compute the unit vector in the direction of the line segment connecting the original points
    ux = (jx - ix) / d
    uy = (jy - iy) / d

    # Compute the perpendicular unit vector to the line segment
    vx = -uy
    vy = -ux

    # Compute the coordinates of the new points
    k = math.sqrt(d ** 2 - (d / 2) ** 2)  # distance from midpoint to new points
    new_ix = round(ix + ux)
    new_iy = round(iy + uy)
    new_jx = round(jx - ux)
    new_jy = round(jy - uy)
    return (new_ix, new_iy, new_jx, new_jy)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def train(X_train, y_train):
    # Define the neural network architecture
    input_dim = 784
    hidden_dim = 32
    output_dim = 10

    # Initialize the weights and biases
    W1 = np.random.randn(input_dim, hidden_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim)
    b2 = np.zeros((1, output_dim))

    # Set the learning rate and number of epochs
    learning_rate = 0.01
    num_epochs = 1000

    # Loop over the training data for the specified number of epochs
    for epoch in range(num_epochs):
        # Forward propagation
        z1 = np.dot(X_train, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = sigmoid(z2)

        # Compute the loss
        loss = np.mean((y_pred - y_train)**2)

        # Backpropagation
        delta3 = (y_pred - y_train) * sigmoid_derivative(z2)
        delta2 = np.dot(delta3, W2.T) * sigmoid_derivative(z1)
        dW2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        dW1 = np.dot(X_train.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Update the weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        # Print the loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss:.4f}")

def pearson_correlation(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum([xi*yi for xi, yi in zip(x, y)])
    sum_x_squared = sum([xi**2 for xi in x])
    sum_y_squared = sum([yi**2 for yi in y])

    covariance = sum_xy/n - (sum_x/n)*(sum_y/n)
    std_dev_x = math.sqrt(sum_x_squared/n - (sum_x/n)**2)
    std_dev_y = math.sqrt(sum_y_squared/n - (sum_y/n)**2)

    correlation = covariance / (std_dev_x * std_dev_y)

    return correlation


def matrix_correlations_v2(X, Y):
    n = X.shape[2]
    C = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i, j, :]
            y = Y[i, j, :]
            covariance = np.sum((x - np.mean(x)) * (y - np.mean(y))) / n
            std_dev_x = np.sqrt(np.sum((x - np.mean(x)) ** 2) / n)
            std_dev_y = np.sqrt(np.sum((y - np.mean(y)) ** 2) / n)
            correlation = covariance / (std_dev_x * std_dev_y)
            C[i,j] = correlation
    return C

def get_gray_image(cap):

    # Capture a frame from the camera
    ret = False

    # Continuously capture frames from the camera until a valid frame is available
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is valid
        if ret:
            break
        #cv2.waitKey(10)

    # Resize the frame to a smaller size
    resized_frame = cv2.resize(frame, CAMERA_DIM)

    # Convert the resized frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Flatten the grayscale frame to a one-dimensional numpy array
    gray_vector = np.ravel(gray_frame) / 255.0

    return gray_vector

def show(gray_vector):
    # Reshape the flattened grayscale frame to its original size
    gray_frame = np.reshape(gray_vector, (CAMERA_DIM[1], CAMERA_DIM[0]))

    # Normalize the grayscale frame to the range of 0 to 255
    normalized_frame = cv2.normalize(
        gray_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the normalized frame as an image
    cv2.imshow("Normalized Frame", normalized_frame)
    cv2.waitKey(1)

def sorter(temporal, W1, input_dim, output_dim):
    i = random.randint(0, input_dim-1)
    j = random.randint(0, output_dim-1)
    ix, iy = coords(i, CAMERA_DIM[0], CAMERA_DIM[1])
    jx, jy = coords(j, CAMERA_DIM[0], CAMERA_DIM[1])


    (new_ix, new_iy, new_jx, new_jy) = get_new_coordinates(ix, iy, jx, jy)
    new_i = to_index(new_ix, new_iy, CAMERA_DIM[0], CAMERA_DIM[1])
    new_j = to_index(new_jx, new_jy, CAMERA_DIM[0], CAMERA_DIM[1])
    #W1[new_i], W1[i] = W1[i], W1[new_i]
    #W1[new_j], W1[j] = W1[j], W1[new_j]
    rate = 0.01
    corr = correlation(temporal, i, j)
    if corr > 0.9:
        dwi = W1[i, j] * rate
        #W1[i, j] -= dwi
        #W1[new_i, new_j] += dwi
    #elif abs(corr) < 0.1:
    #    dwi = W1[i, j] * rate
    #    W1[i, j] = 0.0
    #    #W1[new_i, new_j] -= dwi
    #else:
    #    W1[i, j] = 0.0
    #W1[i, j] = 0

    return True

def sort_weights(temporal_input, temporal_output, W1, input_dim, output_dim):
    i = random.randint(0, input_dim-1)
    j = random.randint(0, output_dim-1)
    rate = 0.01
    corr = correlation(temporal_input, temporal_output, i, j)
    if corr > 0.9:
        print(corr)
        d = rate * (corr)
        W1[i, j] += d
        W1[i] = W1[i] / np.sum(W1[i])

    return True

def sort_output(temporal_input, temporal_output, W1, input_dim, output_dim, skip):
    i = random.randint(0, output_dim-1)
    j = random.randint(0, output_dim-1)
    if i in skip or j in skip:
        return False
    ix, iy = coords(i, CAMERA_DIM[0], CAMERA_DIM[1])
    jx, jy = coords(j, CAMERA_DIM[0], CAMERA_DIM[1])
    #d = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2)
    #if d == 0 or d > 5:
    #    return False
    rate = 0.01
    corr = correlation(temporal_output, temporal_output, i, j)
    if corr > 0.99:
        print(corr)
        (new_ix, new_iy, new_jx, new_jy) = get_new_coordinates(ix, iy, jx, jy)
        new_i = to_index(new_ix, new_iy, CAMERA_DIM[0], CAMERA_DIM[1])
        new_j = to_index(new_jx, new_jy, CAMERA_DIM[0], CAMERA_DIM[1])
        if new_i in skip or new_j in skip:
            return False
        skip[i] = True
        skip[j] = True
        skip[new_i] = True
        skip[new_j] = True
        W1[:, [i, new_i]] = W1[:, [new_i, i]]
        W1[:, [j, new_j]] = W1[:, [new_j, j]]
        #W1[:, [i, new_j]] = W1[:, [new_j, i]]
        #W1[:, [j, new_i]] = W1[:, [new_i, j]]
        return True
    return False

def coords(i, w, h):
    x = i % w
    y = i // w
    return (x, y)

def to_index(x, y, w, h):
    return y*w + x

def correlation(temporal_input, temporal_output, i, j):
    if i == j:
        return 1.0
    x = temporal_input[i, :]
    y = temporal_output[j, :]
    n = x.size
    #if not np.all(np.diff(x)) or not np.all(np.diff(y)):
    #    return 0.0
    #if np.average(np.abs(np.diff(x))) < 0.05:
    #    return 0.0
    covariance = np.sum((x - np.mean(x)) * (y - np.mean(y))) / n
    std_dev_x = np.sqrt(np.sum((x - np.mean(x)) ** 2) / n)
    std_dev_y = np.sqrt(np.sum((y - np.mean(y)) ** 2) / n)
    if not std_dev_x * std_dev_y:
        return 0.0
    correlation = covariance / (std_dev_x * std_dev_y)
    return correlation

def create_weight_matrix(neurons_count):
    weight_matrix = np.zeros((neurons_count, neurons_count))
    for i in range(neurons_count):
        for j in range(i+1, neurons_count):
            weight = 1/abs(i-j)
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight
    return weight_matrix

def cost_function_sampled(weight_matrix, W1, cols, sample_size):
    neurons_count = weight_matrix.shape[0]
    cost = 0
    for _ in range(sample_size):
        i, j = random.sample(range(neurons_count), 2)
        if i != j:
            x1, y1 = i % cols, i // cols
            x2, y2 = j % cols, j // cols
            dist = abs(x1 - x2) + abs(y1 - y2)
            weight = get_neuron_weight(weight_matrix, W1, i, j)
            cost += weight * dist
    return cost

def cost_function(weight_matrix, W1, cols):
    neurons_count = weight_matrix.shape[0]
    cost = 0
    for i in range(neurons_count):
        for j in range(neurons_count):
            if i != j:
                x1, y1 = i % cols, i // cols
                x2, y2 = j % cols, j // cols
                dist = abs(x1 - x2) + abs(y1 - y2)
                weight = get_neuron_weight(weight_matrix, W1, i, j)
                cost += weight * dist
    return cost

def simulated_annealing(weight_matrix, W1, init_temp=100, cooling_rate=0.99, iterations=100, cols=CAMERA_DIM[0], rows=CAMERA_DIM[1]):
    neurons_count = weight_matrix.shape[0]
    #current_cost = cost_function(weight_matrix, W1, cols)
    current_cost = cost_function_sampled(weight_matrix, W1, cols, sample_size = neurons_count*neurons_count // 2)
    temp = init_temp

    for _ in range(iterations):
        i = random.randint(0, neurons_count-1)
        j = random.randint(0, neurons_count-1)
        x1, y1 = coords(i, cols, rows)
        x2, y2 = coords(j, cols, rows)

        W1[:, [i, j]] = W1[:, [j, i]]

        #new_cost = cost_function(weight_matrix, W1, cols)
        new_cost = cost_function_sampled(weight_matrix, W1, cols, sample_size = neurons_count*neurons_count // 2)
        delta_cost = new_cost - current_cost

        if delta_cost < 0 or math.exp(-delta_cost / temp) > random.random():
            current_cost = new_cost
        else:
            W1[:, [i, j]] = W1[:, [j, i]]

        temp *= cooling_rate

def get_neuron(W1, j):
    idx = np.where(W1[:, j] == 1)
    if len(idx[0]) == 0:
        return None
    return idx[0][0]

def get_weight(weight_matrix, i, j):
    return weight_matrix[i, j][0]

def get_neuron_weight(weight_matrix, W1, i, j):
    return weight_matrix[get_neuron(W1, i), get_neuron(W1, j)][0]

def update_weights(weight_matrix, weight, i, j):
    old_weight = weight_matrix[i, j, 0]
    count = weight_matrix[i, j, 1]
    tail = 10
    if count >= tail - 1:
        count = tail - 1
    weight_matrix[i, j, 0] = (old_weight*count + weight)/(count + 1)
    weight_matrix[i, j, 1] = count + 1
    weight_matrix[j, i, 0] = weight_matrix[i, j, 0]
    weight_matrix[j, i, 1] = weight_matrix[i, j, 1]

def sort_output_sa(temporal_input, temporal_output, W1, input_dim, output_dim, weight_matrix):
    i = random.randint(0, output_dim-1)
    j = random.randint(0, output_dim-1)
    corr = correlation(temporal_output, temporal_output, i, j)
    update_weights(weight_matrix, abs(corr), get_neuron(W1, i), get_neuron(W1, j))


def tick():
    steps = 5
    input_dim = np.prod(CAMERA_DIM)
    output_dim = input_dim
    temporal_input = np.zeros((input_dim, steps))
    temporal_output = np.zeros((output_dim, steps))
    # 0 - weight and 1 - count of times it was updated
    #weight_matrix = np.zeros((output_dim, output_dim, 2))
    #W1 = np.eye(input_dim, output_dim)
    #np.random.shuffle(W1)
    #W1 = np.random.rand(input_dim, output_dim)
    #for i in range(32*10):
    #    for j in range(output_dim):
    #        W1[j, i + CAMERA_DIM[0]*5] = 0
    # Initialize the default camera
    cap = cv2.VideoCapture(0)
    tick = 0
    while True:
        step = tick % steps
        tick += 1
        temporal_input[:, step] = get_gray_image(cap)
        input = temporal_input[:, step]
        #z1 = np.dot(input, W1)
        #a1 = sigmoid(z1)
        #temporal_output[:, step] = z1
        success_sort_count = 0
        variance = np.var(temporal_input, axis=1)
        #show(np.select([variance > 0.01], [variance]))
        show(variance)
        #print(max(np.select([variance > 0.01], [variance])))
        #print(max(variance))
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    assert get_new_coordinates(1, 1, 1, 1) == (1, 1, 1, 1)
    assert get_new_coordinates(1, 1, 5, 5) == (2, 2, 4, 4)
    assert get_new_coordinates(5, 5, 1, 1) == (4, 4, 2, 2)
    assert get_new_coordinates(-5, -5, 5, 5) == (-4, -4, 4, 4)
    assert get_new_coordinates(1, 1, 1, 2) == (1, 2, 1, 1)
    assert get_neuron(np.eye(5, 5), 0) == 0
    assert get_neuron(np.eye(5, 5), 4) == 4
    assert get_neuron(np.eye(5, 5), 3) == 3
    assert get_neuron(np.rot90(np.eye(5, 5)), 0) == 4
    assert get_neuron(np.eye(3, 7), 1) == 1
    assert get_neuron(np.eye(3, 7), 3) is None
    tick()
