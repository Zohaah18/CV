import cv2
import pandas as pd
import matplotlib.pyplot as plt

image_path="C:\\Users\\LAPTOOL TECHNOLOGY\\OneDrive\\Desktop\\CV_Lab1\\sukuna.png"

image=cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    image_rgb=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    red=image_rgb[:,:,0]
    green=image_rgb[:,:,1]
    blue=image_rgb[:,:,2]

    red_flat=red.flatten()
    green_flat=green.flatten()
    blue_flat=blue.flatten()

    df=pd.DataFrame(
        {
        "Red": red_flat,
        "Green": green_flat,
        "Blue": blue_flat
        }
    )

    print("First 5 rows:")
    print(df.head())

    print("\nStatistical Summary: ")
    print(df.describe())

    print("Mean:")
    print(df.mean())

    print("\nMinimum:")
    print(df.min())

    print("\nMaximum:")
    print(df.max())

    print("\nStandard Deviation:")
    print(df.std())