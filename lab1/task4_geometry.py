import cv2
import numpy as np
import matplotlib.pyplot as plt

canvas=np.zeros(
    (800,800,3),
    dtype=np.uint8
)
center_x= 800//2
center_y= 800//2

center=(center_x,center_y)

print("Center:",center)
radii=[300,240,180,120,60]

colors=[
    (255,255,255),
    (0,0,255),
    (255,255,255),
    (0,0,255),
    (255,255,255)
]

for radius, color in zip(radii,colors):
    cv2.circle(
        canvas,
        center,
        radius,
        color,
        -1
    )
outer_radius= 300

top_left=(
    center_x-outer_radius,
    center_y-outer_radius
)

bottom_right=(
    center_x+outer_radius,
    center_y+outer_radius
)

cv2.rectangle(
    canvas,
    top_left,
    bottom_right,
    (0,255,0),
    3
)

canvas_rgb=cv2.cvtColor(
    canvas,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8,8))

plt.imshow(canvas_rgb)
plt.axis("off")
plt.title(
    "Concentric Circles",
    fontsize=16
)
plt.show()

print("Bounding Box:")
print("Top Left:", top_left)
print("Bottom Right:", bottom_right)
