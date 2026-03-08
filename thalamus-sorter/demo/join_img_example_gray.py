import cv2
import numpy as np

# Load the images
image1 = cv2.imread('./logo1.png')
image2 = cv2.imread('./logo2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Resize images to the same height if necessary
max_height = max(gray1.shape[0], gray2.shape[0])
total_width = gray1.shape[1] + gray2.shape[1]

# Optionally, add some space between the images
padding = 10  # Adjust padding to your needs
total_width += padding

# Create an empty image with the appropriate size
joined_image = np.zeros((max_height, total_width), dtype='uint8')

# Place the first image on the left
joined_image[:gray1.shape[0], :gray1.shape[1]] = gray1

# Place the second image on the right, with padding if specified
joined_image[:gray2.shape[0], gray1.shape[1]+padding:] = gray2

# Save or display the joined image
cv2.imshow('Joined Image', joined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To save the image
# cv2.imwrite('joined_image.jpg', joined_image)

