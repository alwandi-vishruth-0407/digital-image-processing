
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Image
image = cv2.imread('pictures/istockphoto-849220498-612x612.jpg')
if image is None:
    raise ValueError("Image not found. Check the file path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
b, g, r = cv2.split(image)

# 2. Define Kernels
kernel_1d = np.array([[1, 0, -1]])  # Shape (1, 3) - 2D Matrix
kernel_2d = np.array([[1, 1, 1], 
                      [1, -8, 1], 
                      [1, 1, 1]])

# 3. Apply Convolutions using CV_32F to prevent uint8 underflow
conv_1d = cv2.convertScaleAbs(cv2.filter2D(gray, cv2.CV_32F, kernel_1d))
conv_2d = cv2.convertScaleAbs(cv2.filter2D(gray, cv2.CV_32F, kernel_2d))

# Prepare lists
images = [image_rgb, gray, r, g, b, conv_1d, conv_2d]
titles = ['Original Image', 'Grayscale', 'Red Channel', 'Green Channel', 
          'Blue Channel', '1D Convolution', '2D Convolution']
cmaps = [None, 'gray', 'Reds', 'Greens', 'Blues', 'gray', 'gray']

# 4. Generate Figures One After Another
for title, img, cmap in zip(titles, images, cmaps):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Column 1: Image
    if cmap is None:
        axes[0].imshow(img)
    else:
        axes[0].imshow(img, cmap=cmap)
    axes[0].set_title(title, fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Column 2: Center 5x5 Matrix Sample
    axes[1].axis('off')
    h, w = img.shape[:2]
    sample_matrix = img[h//2 : h//2 + 5, w//2 : w//2 + 5]
    if len(sample_matrix.shape) == 3:
        sample_matrix = sample_matrix[:, :, 0]
        
    table = axes[1].table(cellText=sample_matrix.tolist(), cellLoc='center', loc='center')
    table.scale(1.0, 1.6)
    table.set_fontsize(10)
    axes[1].set_title(f"Center 5x5 Matrix Sample", fontsize=10)
    
    # Column 3: Histogram
    if len(img.shape) == 2:
        axes[2].hist(img.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    else:
        colors = ('r', 'g', 'b')
        for j, col in enumerate(colors):
            axes[2].hist(img[:, :, j].ravel(), bins=256, range=[0, 256], color=col, alpha=0.4)
            
    axes[2].set_title("Histogram", fontsize=10)
    axes[2].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()  # Blocks until closed, then proceeds to the next image