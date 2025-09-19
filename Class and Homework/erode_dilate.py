import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load RGB image
img = cv2.imread('/home/kowsar/Documents/Digital_Image_Processing/DIP_Problems/Images/lily.jpeg')

# Define kernel
kernel = np.ones((5,5), np.uint8)

# Apply Dilation and Erosion directly on RGB image
dilated = cv2.dilate(img, kernel, iterations=1)
eroded = cv2.erode(img, kernel, iterations=1)

# Show results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original RGB")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB))
plt.title("Dilated RGB")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(eroded, cv2.COLOR_BGR2RGB))
plt.title("Eroded RGB")
plt.axis('off')

plt.show()
