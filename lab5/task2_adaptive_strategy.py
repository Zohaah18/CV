import os
import cv2
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

img = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)

block_sizes = [11, 35, 75]
C_values = [2, 10, 20]
methods = {"Mean": cv2.ADAPTIVE_THRESH_MEAN_C, "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

results = {}
for name, flag in methods.items():
    for bs in block_sizes:
        for C in C_values:
            results[(name, bs, C)] = cv2.adaptiveThreshold(img, 255, flag, cv2.THRESH_BINARY, bs, C)

chosen = [("Mean", 11, 2), ("Mean", 35, 10), ("Mean", 75, 20),
          ("Gaussian", 11, 2), ("Gaussian", 35, 10), ("Gaussian", 75, 20)]

fig, ax = plt.subplots(1, len(chosen) + 1, figsize=(24, 4.6))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")
for a, (name, bs, C) in zip(ax[1:], chosen):
    a.imshow(results[(name, bs, C)], cmap="gray")
    a.set_title(f"{name}\nbs={bs}, C={C}", fontsize=9)
    a.axis("off")
    cv2.imwrite(f"outputs/task2_{name}_bs{bs}_C{C}.png", results[(name, bs, C)])
plt.tight_layout()
plt.savefig("outputs/task2_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(len(block_sizes), len(C_values), figsize=(12, 15))
for i, bs in enumerate(block_sizes):
    for j, C in enumerate(C_values):
        ax[i, j].imshow(results[("Gaussian", bs, C)], cmap="gray")
        ax[i, j].set_title(f"Gaussian bs={bs}, C={C}", fontsize=10)
        ax[i, j].axis("off")
plt.tight_layout()
plt.savefig("outputs/task2_gaussian_grid.png", dpi=150, bbox_inches="tight")
plt.show()

print("1. Neighbourhood too small (11): breaks or loses thin digit strokes.")
print("2. Neighbourhood too large (75): shadow contrast re-enters, edges become thicker and blockier.")
print("3. C increased: removes background speckle, but too high thins or drops strokes.")
print("4. Cleanest combination: blockSize=35, C=10.")
print("5. Better method: Gaussian, its centre weighting is less sensitive to stray pixels.")
