#================= Importing necessary libraries ======================
import cv2
import numpy as np
import matplotlib.pyplot as plt

#================= Helper function ======================
def plot_histogram(img):
    """
    Compute histogram, PDF, and CDF for a grayscale image
    """
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).ravel()
    pdf = hist / hist.sum()               # Normalize to probability
    cdf = pdf.cumsum()                    # Cumulative distribution
    cdf_normalized = cdf / cdf.max()      # Normalize CDF to [0,1]
    return hist, pdf, cdf_normalized

#================= Main execution ======================
def main():
    # Load the grayscale image
    img_path = "/home/kowsar/Documents/Digital_Image_Processing/DIP_Problems/Images/lily.jpeg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Image not found or unable to load.")
        return

    # Resize image for safe plotting
    img_disp = cv2.resize(img, (400, 400))

    # 1. Global Histogram Equalization
    hist_eq = cv2.equalizeHist(img)
    hist_eq_disp = cv2.resize(hist_eq, (400, 400))

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_eq = clahe.apply(img)
    clahe_eq_disp = cv2.resize(clahe_eq, (400, 400))

    # Compute histograms, PDFs, CDFs
    hist_orig, pdf_orig, cdf_orig = plot_histogram(img)
    hist_he, pdf_he, cdf_he = plot_histogram(hist_eq)
    hist_clahe, pdf_clahe, cdf_clahe = plot_histogram(clahe_eq)

    #================= Plotting ======================
    plt.figure(figsize=(16, 12))

    # Display images
    plt.subplot(4,3,1)
    plt.imshow(img_disp, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(4,3,2)
    plt.imshow(hist_eq_disp, cmap='gray')
    plt.title("Histogram Equalized")
    plt.axis('off')

    plt.subplot(4,3,3)
    plt.imshow(clahe_eq_disp, cmap='gray')
    plt.title("CLAHE Image")
    plt.axis('off')

    # Display histograms
    plt.subplot(4,3,4)
    plt.plot(hist_orig, color='black')
    plt.title("Original Histogram")

    plt.subplot(4,3,5)
    plt.plot(hist_he, color='blue')
    plt.title("HE Histogram")

    plt.subplot(4,3,6)
    plt.plot(hist_clahe, color='green')
    plt.title("CLAHE Histogram")

    # Display PDFs
    plt.subplot(4,3,7)
    plt.plot(pdf_orig, color='black')
    plt.title("Original PDF")

    plt.subplot(4,3,8)
    plt.plot(pdf_he, color='blue')
    plt.title("HE PDF")

    plt.subplot(4,3,9)
    plt.plot(pdf_clahe, color='green')
    plt.title("CLAHE PDF")

    # Display CDFs
    plt.subplot(4,3,10)
    plt.plot(cdf_orig, color='black')
    plt.title("Original CDF")

    plt.subplot(4,3,11)
    plt.plot(cdf_he, color='blue')
    plt.title("HE CDF")

    plt.subplot(4,3,12)
    plt.plot(cdf_clahe, color='green')
    plt.title("CLAHE CDF")

    plt.tight_layout()
    plt.show()

#================= Run main ======================
if __name__ == "__main__":
    main()
