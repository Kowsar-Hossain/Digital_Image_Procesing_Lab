import matplotlib.pyplot as plt
import cv2
import numpy as np

# Threshold limits
limit1 = 127
limit2 = 200

def main():
    # Read image in grayscale
    img_gray = cv2.imread("/home/kowsar/Documents/Image_Processing/lily.jpeg", 0)

    # Apply thresholding methods
    img1 = thresolding1(img_gray)
    img2 = thresolding2(img_gray)
    img3 = thresolding3(img_gray)

    # Display results with histograms
    display_with_histograms([img_gray, img1, img2, img3], 
                            ["Original", "Thresholding 1", "Thresholding 2", "Thresholding 3"])

def display_with_histograms(images, titles):
    plt.figure(figsize=(12, 8))

    for idx, (img, title) in enumerate(zip(images, titles)):
        # Show image
        plt.subplot(4, 2, 2*idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

        # Show histogram
        plt.subplot(4, 2, 2*idx+2)
        plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray')
        plt.title(f"Histogram - {title}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def thresolding1(img_gray):
    img_tmp = img_gray.copy()
    img_tmp[img_tmp <= limit1] = 0
    img_tmp[img_tmp > limit1] = 255
    return img_tmp

def thresolding2(img_gray):
    img_tmp = img_gray.copy()
    mask = (img_tmp >= limit1) & (img_tmp <= limit2)
    img_tmp[mask] = 127
    return img_tmp

def thresolding3(img_gray):
    img_tmp = img_gray.copy()
    mask_mid = (img_tmp >= limit1) & (img_tmp <= limit2)
    mask_high = img_tmp > limit2
    img_tmp[mask_mid] = 127
    img_tmp[mask_high] = 255
    return img_tmp

if __name__ == '__main__':
    main()
