'''
Problem Statement (Histogram Equalization with Reference Image):
Given an original grayscale image and a reference grayscale image,
perform histogram equalization using reference image CDF.
Plot original, reference, and enhanced images along with their
histograms and CDFs.
'''

#================= Importing necessary libraries ======================
import matplotlib.pyplot as plt
import numpy as np
import cv2


#================= Execution workflow ==============
def main():
    # --- Input paths (change if needed) ---
    img_path = "/home/kowsar/Documents/Digital_Image_Processing/DIP_Problems/Images/lily.jpeg"
    ref_path = "/home/kowsar/Documents/Digital_Image_Processing/DIP_Problems/Images/color.jpeg"

    # --- Read images in grayscale ---
    img_gray = cv2.imread(img_path, 0)
    ref_gray = cv2.imread(ref_path, 0)

    # --- Check if images are loaded ---
    if img_gray is None:
        raise FileNotFoundError(f"Could not load original image from {img_path}")
    if ref_gray is None:
        raise FileNotFoundError(f"Could not load reference image from {ref_path}")

    # --- Histogram & CDF for original ---
    hist_orig = histogram(img_gray)
    pdf_orig = pdf_f(hist_orig)
    cdf_orig = cdf_f(pdf_orig)

    # --- Histogram & CDF for reference ---
    hist_ref = histogram(ref_gray)
    pdf_ref = pdf_f(hist_ref)
    cdf_ref = cdf_f(pdf_ref)

    # --- Histogram Equalization using reference ---
    cdf_min = cdf_ref[cdf_ref > 0].min()     # minimum non-zero CDF from reference
    L = 256
    new_level = np.round(((cdf_ref - cdf_min) / (1 - cdf_min)) * (L - 1)).astype(np.uint8)

    enhanced_img = img_conv(img_gray, new_level)

    # --- Histogram & CDF for enhanced ---
    hist_enh = histogram(enhanced_img)
    pdf_enh = pdf_f(hist_enh)
    cdf_enh = cdf_f(pdf_enh)

    #================= Display =================
    display_results(img_gray, ref_gray, enhanced_img,
                    hist_orig, cdf_orig,
                    hist_ref, cdf_ref,
                    hist_enh, cdf_enh)


#================= Apply new intensity levels to image ================
def img_conv(img_gray, new_level):
    return new_level[img_gray]


#================= Function to calculate histogram ====================
def histogram(img_2D):
    h, w = img_2D.shape
    hist = np.zeros(256, dtype=int)

    for i in range(h):
        for j in range(w):
            pixel_value = img_2D[i, j]
            hist[pixel_value] += 1

    return hist


#================= Function to calculate PDF ==========================
def pdf_f(hist):
    return hist / hist.sum()


#================= Function to calculate CDF ==========================
def cdf_f(pdf):
    return np.cumsum(pdf)


#================= Display images and histograms ======================
def display_results(img_orig, img_ref, img_enh,
                    hist_orig, cdf_orig,
                    hist_ref, cdf_ref,
                    hist_enh, cdf_enh):

    plt.figure(figsize=(18, 12))

    # ---- Show Images ----
    plt.subplot(3, 3, 1)
    plt.imshow(img_orig, cmap="gray")
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(img_ref, cmap="gray")
    plt.title("Reference Image")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(img_enh, cmap="gray")
    plt.title("Enhanced Image")
    plt.axis('off')

    # ---- Show Histograms ----
    plt.subplot(3, 3, 4)
    plt.bar(range(256), hist_orig)
    plt.title("Original Histogram")

    plt.subplot(3, 3, 5)
    plt.bar(range(256), hist_ref)
    plt.title("Reference Histogram")

    plt.subplot(3, 3, 6)
    plt.bar(range(256), hist_enh)
    plt.title("Enhanced Histogram")

    # ---- Show CDFs ----
    plt.subplot(3, 3, 7)
    plt.plot(cdf_orig, color='b')
    plt.title("Original CDF")

    plt.subplot(3, 3, 8)
    plt.plot(cdf_ref, color='g')
    plt.title("Reference CDF")

    plt.subplot(3, 3, 9)
    plt.plot(cdf_enh, color='r')
    plt.title("Enhanced CDF")

    plt.tight_layout()
    plt.savefig("result.png")  # Save output as image file
    plt.show()


#================= Main function to run the script ====================
if __name__ == "__main__":
    main()
