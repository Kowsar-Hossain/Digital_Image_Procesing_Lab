import cv2
import numpy as np

# Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===============================
    # 1️⃣ Linear Mapping (Contrast Stretching)
    # g(x) = alpha*f(x) + beta
    # ===============================
    alpha = 1.5   # Contrast
    beta = 30     # Brightness
    linear_img = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # ===============================
    # 2️⃣ Nonlinear Mapping (Gamma Correction)
    # g(x) = 255*(f(x)/255)^gamma
    # ===============================
    gamma = 0.5
    invGamma = 1.0 / gamma

    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    nonlinear_img = cv2.LUT(gray, table)

    # ===============================
    # Histogram Calculation
    # ===============================
    hist_orig = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist_linear = cv2.calcHist([linear_img], [0], None, [256], [0,256])
    hist_nonlinear = cv2.calcHist([nonlinear_img], [0], None, [256], [0,256])

    # Normalize histogram for display
    hist_orig = cv2.normalize(hist_orig, None, 0, 300, cv2.NORM_MINMAX).flatten()
    hist_linear = cv2.normalize(hist_linear, None, 0, 300, cv2.NORM_MINMAX).flatten()
    hist_nonlinear = cv2.normalize(hist_nonlinear, None, 0, 300, cv2.NORM_MINMAX).flatten()

    # Create histogram image
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)

    for x in range(256):
        cv2.line(hist_img, (x,300), (x,300-int(hist_orig[x])), (255,255,255), 1)
        cv2.line(hist_img, (x,300), (x,300-int(hist_linear[x])), (255,0,0), 1)
        cv2.line(hist_img, (x,300), (x,300-int(hist_nonlinear[x])), (0,255,0), 1)

    # Convert grayscale to BGR for stacking
    linear_col = cv2.cvtColor(linear_img, cv2.COLOR_GRAY2BGR)
    nonlinear_col = cv2.cvtColor(nonlinear_img, cv2.COLOR_GRAY2BGR)

    # Resize images
    original_resized = cv2.resize(frame, (320,240))
    linear_resized = cv2.resize(linear_col, (320,240))
    nonlinear_resized = cv2.resize(nonlinear_col, (320,240))

    # Stack horizontally
    top_row = np.hstack((original_resized,
                         linear_resized,
                         nonlinear_resized))

    # Resize histogram
    hist_resized = cv2.resize(hist_img, (960,240))

    # Stack vertically
    final_frame = np.vstack((top_row, hist_resized))

    cv2.imshow("Original | Linear | Nonlinear + Histogram", final_frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
