import numpy as np

height = 64  # Height of the image
width = 64   # Width of the image

# Initialize L1 with random pixel values between 0 and 255
L1 = np.random.randint(0, 256, (height, width))

# Initialize W1 with random connections (one-to-one connections between L1 and L2)
W1 = np.zeros((height * width, height * width))

for i in range(height * width):
    j = np.random.randint(height * width)
    while W1[i, j] == 1.0:
        j = np.random.randint(height * width)
    W1[i, j] = 1.0

np.dot(L1, W1)
