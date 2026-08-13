import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image in grayscale
image_path = 'pictures\images.jpg'
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise ValueError("Image not found. Check the file path.")

# 2. Perform Bit Plane Slicing efficiently using bit-shift operations
# Bit 7 is the Most Significant Bit (MSB), Bit 0 is the Least Significant Bit (LSB)
bit_planes = [((gray >> i) & 1) * 255 for i in range(8)]
bit_planes_uint8 = [img.astype(np.uint8) for img in bit_planes]

# Prepare visualization data (Grayscale + 8 Bit Planes)
images = [gray] + bit_planes_uint8
titles = ['Grayscale Image'] + [f'Bit Plane {i}' for i in range(8)]

# 3. Display Figures One After Another (Clean & Unsquished)
for title, img in zip(titles, images):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # --- Column 1: Image (Natural Aspect Ratio) ---
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(title, fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # --- Column 2: Center 5x5 Matrix Sample ---
    axes[1].axis('off')
    h, w = img.shape
    sample_matrix = img[h//2 : h//2 + 5, w//2 : w//2 + 5]
    
    table = axes[1].table(cellText=sample_matrix.tolist(), cellLoc='center', loc='center')
    table.scale(1.0, 1.6)
    table.set_fontsize(10)
    axes[1].set_title("Center 5x5 Matrix Sample", fontsize=10, pad=12)
    
    # --- Column 3: Histogram ---
    axes[2].hist(img.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    axes[2].set_title("Histogram", fontsize=10, pad=10)
    axes[2].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()  # Close each figure (X) to proceed to the next bit plane