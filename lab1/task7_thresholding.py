import cv2
import matplotlib.pyplot as plt

image_path=r"C:\Users\LAPTOOL TECHNOLOGY\OneDrive\Desktop\CV_Lab1\sukuna.png"

gray=cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if gray is None:
    print("Error: Document image not found.")
else:
    _, global_binary= cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    adaptive= cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    height , width =adaptive.shape

    center=(
        width//2,
        height//2
    )

    rotation_matrix=cv2.getRotationMatrix2D(
        center,
        45,
        0.8
    )

    rotated=cv2.warpAffine(
        adaptive,
        rotation_matrix,
        (width,height)
    )

    fig, axes=plt.subplots(
        1,
        3,
        figsize=(15,5)
    )

    axes[0].imshow(
        global_binary,
        cmap="gray"
    )

    axes[0].axis("off")

    axes[0].set_title(
        "Global Binary Threshold."
    )

    axes[1].imshow(
        adaptive,
        cmap="gray"
    )

    axes[1].axis("off")
    axes[1].set_title(
        "Adaptive Threshold",
        fontsize=14,
        color="darkgreen"
    )

    axes[2].imshow(
        rotated,
        cmap="gray"
    )

    axes[2].axis("off")
    axes[2].set_title(
        "Adaptive + 45° Rotation",
        fontsize=14,
        color="darkred"
    )

    plt.tight_layout()
    plt.show()