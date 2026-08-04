
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and Resize Images
img1 = cv2.imread('pictures/pink flower.png')
img2 = cv2.imread('pictures/flower.jpg')

if img1 is None or img2 is None:
    raise ValueError("One or both images could not be loaded. Check file paths.")

img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 2. Normalize to [0, 1] for floating point arithmetic
img1_norm = img1.astype(np.float32) / 255.0
img2_norm = img2.astype(np.float32) / 255.0

# --- Arithmetic Operations ---
add = np.clip(cv2.add(img1_norm, img2_norm) * 255.0, 0, 255).astype(np.uint8)
subtract = np.clip(cv2.subtract(img1_norm, img2_norm) * 255.0, 0, 255).astype(np.uint8)
multiply = np.clip(cv2.multiply(img1_norm, img2_norm) * 255.0, 0, 255).astype(np.uint8)
divide = np.clip(cv2.divide(img1_norm, img2_norm + 1e-5) * 255.0, 0, 255).astype(np.uint8)

# --- Bitwise Operations ---
bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_xor = cv2.bitwise_xor(img1, img2)
bitwise_not_img1 = cv2.bitwise_not(img1)
bitwise_not_img2 = cv2.bitwise_not(img2)

titles = [
    'Original Image 1', 'Original Image 2', 'Addition', 'Subtraction',
    'Multiplication', 'Division', 'Bitwise AND', 'Bitwise OR',
    'Bitwise XOR', 'Bitwise NOT (Img 1)', 'Bitwise NOT (Img 2)'
]

images = [
    img1, img2, add, subtract, multiply, divide,
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not_img1, bitwise_not_img2
]

# 3. Generate Figures One After Another
for title, img in zip(titles, images):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Column 1: Image (BGR -> RGB)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title, fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Column 2: Center 5x5 Matrix (Channel 0)
    axes[1].axis('off')
    h, w = img.shape[:2]
    sample_matrix = img[h//2 : h//2 + 5, w//2 : w//2 + 5, 0]
    
    table = axes[1].table(cellText=sample_matrix.tolist(), cellLoc='center', loc='center')
    table.scale(1.0, 1.6)
    table.set_fontsize(10)
    axes[1].set_title("Center 5x5 Matrix (Ch 0)", fontsize=10)
    
    # Column 3: Histograms
    colors = ('r', 'g', 'b')
    for j, col in enumerate(colors):
        axes[2].hist(img[:, :, j].ravel(), bins=256, range=[0, 256], color=col, alpha=0.4)
    axes[2].set_title("Histogram", fontsize=10)
    axes[2].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()  # Blocks until closed, then proceeds to the next image