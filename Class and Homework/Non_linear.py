import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image in grayscale
img = cv2.imread('/home/kowsar/Documents/Image_Processing/DIP_Problems/download.jpeg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Image not found")
    exit()

# Normalize image to [0,1]
img_normalized = img / 255.0

# Gamma values to apply
gamma_values = [0.1, 0.3, 0.5, 1.0]

# Prepare the plot
plt.figure(figsize=(12, 8))

# Show original image
plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

# Apply and show gamma corrected images
for idx, g in enumerate(gamma_values):
    gamma_corrected = np.power(img_normalized, g)
    corrected_img = np.uint8(gamma_corrected * 255)

    plt.subplot(2, 3, idx + 2)
    plt.title(f"Gamma {g}")
    plt.imshow(corrected_img, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
