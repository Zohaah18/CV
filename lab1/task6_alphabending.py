import cv2
import matplotlib.pyplot as plt
image_path="C:\\Users\\LAPTOOL TECHNOLOGY\\OneDrive\\Desktop\\CV_Lab1\\sukuna.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    height,width=image.shape[:2]

    overlay=image.copy()
    start_y= int(height*0.80)

    cv2.rectangle(
        overlay,
        (0,start_y),
        (width,height),
        (255,0,0),
        -1
    )

    blended= cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    cv2.putText(
        blended,
        "KING OF CURSES",
        (30,height-40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255,255,255),
        3,
        cv2.LINE_AA
    )

    blended_rgb=cv2.cvtColor(
        blended,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(10,7))
    plt.imshow(blended_rgb)
    plt.axis("off")
    plt.title(
        "Alpha Blending and Typograhy",
        fontsize=16
    )

    plt.show()
    