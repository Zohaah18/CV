import os
import cv2
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

img = cv2.imread("smarties.png", cv2.IMREAD_GRAYSCALE)

pairs = [(30, 90), (30, 150), (80, 150)]
edges = [cv2.Canny(img, lo, hi) for lo, hi in pairs]

fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")
for i, (a, e, (lo, hi)) in enumerate(zip(ax[1:], edges, pairs), 1):
    a.imshow(e, cmap="gray")
    a.set_title(f"Edge Result {i} (low={lo}, high={hi})")
    a.axis("off")
    cv2.imwrite(f"outputs/task5_edges_low{lo}_high{hi}.png", e)
plt.tight_layout()
plt.savefig("outputs/task5_required_output.png", dpi=150, bbox_inches="tight")
plt.show()

print("Strong edges: outer circular boundary of every candy, present in all three results.")
print("Weak edges: specular highlights and internal shading, present only when low=30.")
print("Missing edges: internal highlights are lost at low=80, high=150.")
print("Unwanted edges: faint background texture appears at low=30, high=90.")
print()
print("Effect of the thresholds: the high threshold sets which edges are accepted as definite, so raising")
print("it drops faint edges. The low threshold sets which connected pixels hysteresis can recover, so")
print("raising it removes noise but also loses weak-but-real edges.")
