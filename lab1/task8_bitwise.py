import cv2
import numpy as np
import matplotlib.pyplot as plt

image1_path = r"C:\Users\LAPTOOL TECHNOLOGY\OneDrive\Desktop\CV_Lab1\sukuna.png"
image2_path = r"C:\Users\LAPTOOL TECHNOLOGY\OneDrive\Desktop\CV_Lab1\sukuna2.png"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

if image1 is None:
    print("Error: First image not found.")

elif image2 is None:
    print("Error: Second image not found.")

else:
    image1 = cv2.resize(image1, (500, 500))
    image2 = cv2.resize(image2, (500, 500))

    mask = np.zeros(
        (500, 500),
        dtype=np.uint8
    )

    polygon = np.array([
        [100, 100],
        [400, 100],
        [450, 250],
        [400, 400],
        [100, 400],
        [50, 250]
    ], dtype=np.int32)

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    foreground = cv2.bitwise_and(
        image1,
        image1,
        mask=mask
    )

    inverted_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(
        image2,
        image2,
        mask=inverted_mask
    )

    result = cv2.bitwise_or(
        foreground,
        background
    )
    image1_rgb = cv2.cvtColor(
        image1,
        cv2.COLOR_BGR2RGB
    )

    image2_rgb = cv2.cvtColor(
        image2,
        cv2.COLOR_BGR2RGB
    )

    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    axes[0].imshow(image1_rgb)
    axes[0].axis("off")
    axes[0].set_title(
        "Image 1",
        fontsize=14
    )

    axes[1].imshow(mask, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title(
        "Binary Mask",
        fontsize=14
    )

    axes[2].imshow(image2_rgb)
    axes[2].axis("off")
    axes[2].set_title(
        "Image 2",
        fontsize=14
    )

    axes[3].imshow(result_rgb)
    axes[3].axis("off")
    axes[3].set_title(
        "Final Bitwise Result",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()