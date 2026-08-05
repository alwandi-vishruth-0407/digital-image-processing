
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load grayscale image
image = cv2.imread('pictures\images.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Error: Image could not be loaded. Check the file path.")

# --- Precompute Intensity Transformations ---
r_values = np.arange(256, dtype=np.float32)

# Identity
map_identity = r_values.astype(np.uint8)

# Negative
map_negative = (255.0 - r_values).astype(np.uint8)

# Log Transformation
c = 255.0 / np.log(1.0 + 255.0)
map_log = np.clip(c * np.log1p(r_values), 0, 255).astype(np.uint8)

# Power-Law (Gamma) Transformation
gamma = 0.5
map_gamma = np.clip(((r_values / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)

# Contrast Stretching (Corrected Math)
r1, s1, r2, s2 = 50.0, 0.0, 200.0, 255.0
contrast_list = []
for r in range(256):
    r_f = float(r)
    if r_f < r1:
        val = (s1 / r1) * r_f if r1 != 0 else 0.0
    elif r_f <= r2:
        val = ((s2 - s1) / (r2 - r1)) * (r_f - r1) + s1
    else:
        val = ((255.0 - s2) / (255.0 - r2)) * (r_f - r2) + s2 if r2 != 255.0 else 255.0
    contrast_list.append(val)
map_contrast = np.clip(np.array(contrast_list), 0, 255).astype(np.uint8)

# --- Apply Transformations ---
img_negative = cv2.LUT(image, map_negative)
img_log = cv2.LUT(image, map_log)
img_gamma = cv2.LUT(image, map_gamma)
img_contrast = cv2.LUT(image, map_contrast)

transformations = [
    ("Original", image, "$s = r$"),
    ("Negative", img_negative, "$s = 255 - r$"),
    ("Log Transform", img_log, r"$s = c \cdot \log(1 + r)$"),
    ("Gamma Correction", img_gamma, r"$s = c \cdot r^{\gamma}\ (\gamma=0.5)$"),
    ("Contrast Stretching", img_contrast, "Piecewise Linear")
]

# --- Option A: Generate Individual Figure Windows for Each Transformation ---
# This keeps the original proportions intact and allows clean printing/saving page-by-page.

for name, img, formula in transformations:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Column 1: Correct Aspect Ratio Image (No Stretching)
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)  # Preserves natural image aspect ratio
    axes[0].set_title(f"{name}\n{formula}", fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Column 2: Center 5x5 Matrix Patch
    axes[1].axis('off')
    h, w = img.shape
    patch_5x5 = img[h//2 : h//2 + 5, w//2 : w//2 + 5]
    
    table = axes[1].table(cellText=patch_5x5.tolist(), loc='center', cellLoc='center')
    table.scale(1.0, 1.6)
    table.set_fontsize(10)
    axes[1].set_title(f"Center 5x5 Patch ({name})", fontsize=10, pad=12)
    
    # Column 3: Histogram
    axes[2].hist(img.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    axes[2].set_title(f"Histogram ({name})", fontsize=10, pad=10)
    axes[2].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()