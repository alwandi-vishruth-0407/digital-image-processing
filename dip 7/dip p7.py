import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image in grayscale
img_path = 'pictures\salt and pepper noise.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found. Check the file path.")

# 2. Define Kernels and Structuring Elements
kernel_3x3 = np.ones((3, 3), np.uint8)

# Weighted Average Filter Kernel (Low Pass)
weighted_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0

# High Pass Kernels (Laplacian & Sobel)
laplacian_kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

# 3. Apply Spatial Domain Filters
filters = [
    ("Original Image", img),
    
    # --- Spatial Low Pass Filters (Smoothing) ---
    ("Mean Filter (Box Blur)", cv2.blur(img, (5, 5))),
    ("Weighted Avg Filter", cv2.filter2D(img, -1, weighted_kernel)),
    ("Median Filter", cv2.medianBlur(img, 5)),
    ("Min Filter (Erosion)", cv2.erode(img, kernel_3x3)),
    ("Max Filter (Dilation)", cv2.dilate(img, kernel_3x3)),
    
    # --- Spatial High Pass Filters (Edge Detection / Sharpening) ---
    ("Laplacian High Pass", cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_32F, laplacian_kernel))),
    ("Sobel X High Pass", cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3))),
    ("Sobel Y High Pass", cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)))
]

# 4. Display Figures One After Another (Clean & Unsquished)
for name, filtered in filters:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # --- Column 1: Filtered Image (Natural Aspect Ratio) ---
    axes[0].imshow(filtered, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(name, fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # --- Column 2: Center 5x5 Matrix Sample ---
    axes[1].axis('off')
    h, w = filtered.shape
    sample_matrix = filtered[h//2 : h//2 + 5, w//2 : w//2 + 5]
    
    table = axes[1].table(cellText=sample_matrix.tolist(), cellLoc='center', loc='center')
    table.scale(1.0, 1.6)
    table.set_fontsize(10)
    axes[1].set_title("Center 5x5 Matrix Sample", fontsize=10, pad=12)
    
    # --- Column 3: Histogram ---
    axes[2].hist(filtered.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    axes[2].set_title("Histogram", fontsize=10, pad=10)
    axes[2].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()  # Close window (X) to proceed to the next filter output