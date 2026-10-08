import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img_path = "pictures\salt and pepper noise.jpg"
image = cv2.imread(img_path, 0)

if image is None:
    raise ValueError("Image not found. Check the file path.")

# Apply median filter
denoised = cv2.medianBlur(image, 5)

titles = ["Noisy Image", "Denoised Image"]
images = [image, denoised]

# Create figure
fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 10)
)

for i, (img, title) in enumerate(zip(images, titles)):

    # ---------------- IMAGE ----------------
    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title(title, fontsize=14)
    axes[i, 0].axis("off")

    # ---------------- SAMPLE MATRIX ----------------
    axes[i, 1].axis("off")

    # Take a 5 × 5 matrix from the center
    h, w = img.shape
    sample_matrix = img[
        h // 2 - 2:h // 2 + 3,
        w // 2 - 2:w // 2 + 3
    ]

    table = axes[i, 1].table(
        cellText=sample_matrix.tolist(),
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(2, 2)

    axes[i, 1].set_title(
        "Sample Matrix",
        fontsize=14
    )

    # ---------------- HISTOGRAM ----------------
    axes[i, 2].hist(
        img.ravel(),
        bins=256,
        range=[0, 256],
        color="black",
        alpha=0.7
    )

    axes[i, 2].set_title(
        "Histogram",
        fontsize=14
    )

    axes[i, 2].set_xlabel("Pixel Intensity")
    axes[i, 2].set_ylabel("Frequency")
    axes[i, 2].set_xlim(0, 256)

# Overall layout
plt.tight_layout()
plt.show()