import cv2
import matplotlib.pyplot as plt

image_path="C:\\Users\\LAPTOOL TECHNOLOGY\\OneDrive\\Desktop\\CV_Lab1\\sukuna.png"

image=cv2.imread(image_path)

if image is None:
    print("Error: Image could not be loaded.")
    print("Check the image path.")
else:
    print("Image loaded successfully.")

    image_rgb=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    plt.figure(figsize=(8,6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(
        "King of Curses - Ryomen Sukuna",
        fontsize=16
    )
    plt.show()