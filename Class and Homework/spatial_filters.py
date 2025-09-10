import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_for_display(img):
    """Convert filter response to 8-bit displayable format."""
    img_abs = np.abs(img)
    img_norm = cv2.normalize(img_abs, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8)


def main():
    # ==============================
    # 1. Input image (CHANGE HERE)
    # ==============================
    input_path = "lily.jpeg"   # <<---- Boss, put your image name here
    src_color = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if src_color is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")

    # Convert to grayscale
    src = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)

    # ==============================
    # 2. Average (smoothing) filter
    # ==============================
    k_avg = np.ones((3, 3), dtype=np.float32) / 9.0
    avg_filtered = cv2.filter2D(src, ddepth=-1, kernel=k_avg)

    # ==============================
    # 3. Sobel filters
    # ==============================
    k_sobel_x = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)
    k_sobel_y = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32)

    sx = cv2.filter2D(src.astype(np.float32), ddepth=-1, kernel=k_sobel_x)
    sy = cv2.filter2D(src.astype(np.float32), ddepth=-1, kernel=k_sobel_y)

    sobel_mag = np.sqrt(np.square(sx) + np.square(sy))

    sx_disp = normalize_for_display(sx)
    sy_disp = normalize_for_display(sy)
    sobel_mag_disp = normalize_for_display(sobel_mag)

    # ==============================
    # 4. Prewitt filters
    # ==============================
    k_prewitt_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)
    k_prewitt_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]], dtype=np.float32)

    px = cv2.filter2D(src.astype(np.float32), ddepth=-1, kernel=k_prewitt_x)
    py = cv2.filter2D(src.astype(np.float32), ddepth=-1, kernel=k_prewitt_y)

    prewitt_mag = np.sqrt(np.square(px) + np.square(py))

    px_disp = normalize_for_display(px)
    py_disp = normalize_for_display(py)
    prewitt_mag_disp = normalize_for_display(prewitt_mag)

    # ==============================
    # 5. Laplacian filter
    # ==============================
    k_lap = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float32)
    lap = cv2.filter2D(src.astype(np.float32), ddepth=-1, kernel=k_lap)
    lap_disp = normalize_for_display(lap)

    # ==============================
    # 6. Visualization (one merged image)
    # ==============================
    titles = ["Original", "Average", "Sobel X", "Sobel Y", "Sobel Mag",
              "Prewitt X", "Prewitt Y", "Prewitt Mag", "Laplacian"]

    imgs = [src,
            avg_filtered,
            sx_disp,
            sy_disp,
            sobel_mag_disp,
            px_disp,
            py_disp,
            prewitt_mag_disp,
            lap_disp]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, im, t in zip(axes.ravel(), imgs, titles):
        ax.imshow(im, cmap='gray')
        ax.set_title(t, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("merged_result.png", dpi=150)
    plt.close(fig)

    print("✅ Done! Single output saved as: merged_result.png")


if __name__ == "__main__":
    main()
