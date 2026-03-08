import cv2
import math
import numpy as np
import random

CAMERA_DIM = (64//4, 48//4)

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
    #gray_vector = gray_frame / 255.0

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



def total_variation_loss(Y):
    Y = np.reshape(Y, (CAMERA_DIM[1], CAMERA_DIM[0]))
    horizontal_diff = np.abs(Y[:-1, :] - Y[1:, :]).sum()
    vertical_diff = np.abs(Y[:, :-1] - Y[:, 1:]).sum()
    return horizontal_diff + vertical_diff

def compute_gradient(Y, W1):
    Y = np.reshape(Y, (CAMERA_DIM[1], CAMERA_DIM[0]))
    horizontal_diff = Y[:-1, :] - Y[1:, :]
    vertical_diff = Y[:, :-1] - Y[:, 1:]

    dLdY = np.zeros_like(Y)
    dLdY[:-1, :] += horizontal_diff
    dLdY[1:, :] -= horizontal_diff
    dLdY[:, :-1] += vertical_diff
    dLdY[:, 1:] -= vertical_diff
    dLdY = np.reshape(dLdY, (CAMERA_DIM[1], CAMERA_DIM[0]))

    dLdW1 = np.dot(np.ravel(dLdY), W1.T)
    return dLdW1

def optimize_weights(L1, W1, learning_rate, epochs):
    for epoch in range(epochs):
        Y = np.dot(W1, L1) # Forward pass
        loss = total_variation_loss(Y)
        #print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")

        gradient = compute_gradient(Y, W1)
        W1 -= learning_rate * gradient # Update weights
    return Y




def tick():
    learning_rate = 0.001
    epochs = 1000
    input_dim = np.prod(CAMERA_DIM)
    output_dim = input_dim
    cap = cv2.VideoCapture(0)
    tick = 0
    W1 = np.eye(input_dim, output_dim)
    np.random.shuffle(W1)
    while True:
        tick += 1
        L1 = get_gray_image(cap)
        Y = optimize_weights(L1, W1, learning_rate, epochs)
        show(Y)

        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
   tick()
