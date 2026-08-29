import cv2
import matplotlib.pyplot as plt


image_path = r"C:\Users\LAPTOOL TECHNOLOGY\OneDrive\Desktop\CV_Lab1\sukuna.png"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found.")

else:
    height, width = image.shape[:2]

    print("Original image size:", width, "x", height)
    if width < 300 or height < 300:

        print("Image is smaller than 300 x 300.")
        print("Resizing image to 600 x 600.")

        image = cv2.resize(image, (600, 600))

    height, width = image.shape[:2]

    print("Image size used:", width, "x", height)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    blurred = cv2.GaussianBlur(
        image,
        (25, 25),
        0
    )

    blurred_rgb = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2RGB
    )
    center_x = width // 2
    center_y = height // 2

    print("Center:", (center_x, center_y))

    roi_size = 300
    half = roi_size // 2

    x1 = center_x - half
    x2 = center_x + half

    y1 = center_y - half
    y2 = center_y + half

    original_roi = image_rgb[y1:y2, x1:x2]

    blurred_roi = blurred_rgb[y1:y2, x1:x2]

    print("ROI coordinates:")
    print("x:", x1, "to", x2)
    print("y:", y1, "to", y2)

    print("ROI size:", original_roi.shape[1], "x", original_roi.shape[0])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    axes[0].imshow(original_roi)
    axes[0].axis("off")
    axes[0].set_title(
        "Original Center ROI",
        fontsize=14
    )

    axes[1].imshow(blurred_roi)
    axes[1].axis("off")
    axes[1].set_title(
        "Blurred Center ROI",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()