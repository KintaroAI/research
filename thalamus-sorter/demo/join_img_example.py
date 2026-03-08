import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('./logo1.png')
image2 = cv2.imread('./logo2.jpg')

# Determine the new image dimensions
total_width = image1.shape[1] + image2.shape[1] + 10  # Adding 10 pixels of space between images
max_height = max(image1.shape[0], image2.shape[0])

# Create a new image with the determined dimensions, filled with zeros (black)
# Use the same depth and number of channels as the original images
new_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

# Copy the first image into the new image
new_image[:image1.shape[0], :image1.shape[1]] = image1

# Copy the second image into the new image, right next to the first one with some space in between
new_image[:image2.shape[0], image1.shape[1]+10:image1.shape[1]+10+image2.shape[1]] = image2

# Display the result
cv2.imshow('Joined Images', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
