import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Helper function to apply Frequency Domain Filters (Ideal, Gaussian, Butterworth)
def apply_filter(img, f_type, f_name, cutoff, order=2):
    # Construct distance meshgrid relative to frequency center
    u, v = np.meshgrid(
        np.arange(img.shape[1]) - img.shape[1] // 2,
        np.arange(img.shape[0]) - img.shape[0] // 2
    )
    D = np.sqrt(u**2 + v**2)

    # Calculate transfer function H(u, v)
    if f_name == 'ideal':
        H = (D <= cutoff).astype(np.float32)
    elif f_name == 'gaussian':
        H = np.exp(-(D**2) / (2 * (cutoff**2)))
    elif f_name == 'butterworth':
        H = 1 / (1 + (D / (cutoff + 1e-5))**(2 * order))
    else:
        raise ValueError("Unknown filter name")

    # Invert filter for High Pass
    H = 1 - H if f_type == 'highpass' else H

    # Perform 2D FFT, Shift, Filter, Inverse Shift, and Inverse FFT
    F = np.fft.fftshift(np.fft.fft2(img))
    G = F * H
    filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
    
    return np.uint8(np.clip(filtered_img, 0, 255))


# 2. Load the image in grayscale
img_path = 'pictures\salt and pepper noise.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found. Check the file path.")

# 3. Define Filter Configurations
filters = [
    ('lowpass', 'ideal', 'Ideal Low Pass', 20),
    ('lowpass', 'gaussian', 'Gaussian Low Pass', 40),
    ('lowpass', 'butterworth', 'Butterworth Low Pass', 30),
    ('highpass', 'ideal', 'Ideal High Pass', 20),
    ('highpass', 'gaussian', 'Gaussian High Pass', 40),
    ('highpass', 'butterworth', 'Butterworth High Pass', 30)
]

# Compute filtered results
filtered_images = [img]
filtered_titles = ['Original Image']

for f_type, f_name, title, cutoff in filters:
    filtered_images.append(apply_filter(img, f_type, f_name, cutoff))
    filtered_titles.append(title)

# 4. Display Figures One After Another (Clean & Unsquished)
for title, f_img in zip(filtered_titles, filtered_images):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # --- Column 1: Filtered Image ---
    axes[0].imshow(f_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(title, fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # --- Column 2: Center 5x5 Matrix Sample ---
    axes[1].axis('off')
    h, w = f_img.shape
    sample_matrix = f_img[h//2 : h//2 + 5, w//2 : w//2 + 5]
    
    table = axes[1].table(cellText=sample_matrix.tolist(), cellLoc='center', loc='center')
    table.scale(1.0, 1.6)
    table.set_fontsize(10)
    axes[1].set_title("Center 5x5 Matrix Sample", fontsize=10, pad=12)
    
    # --- Column 3: Histogram ---
    axes[2].hist(f_img.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    axes[2].set_title("Histogram", fontsize=10, pad=10)
    axes[2].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.show()  # Close window (X) to proceed to the next filter output