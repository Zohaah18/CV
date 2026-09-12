import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('output', exist_ok=True)
np.set_printoptions(precision=3, suppress=True)

def make_city(w=90, h=70):
    """Synthetic overhead city: a grid of block 'buildings' + a couple of roads."""
    img = np.full((h, w, 3), (40, 90, 40), dtype=np.uint8)
    rng = np.random.default_rng(7)
    for gy in range(6, h - 6, 12):
        for gx in range(6, w - 6, 12):
            bw, bh = int(rng.integers(6, 10)), int(rng.integers(6, 10))
            shade = int(rng.integers(140, 230))
            cv2.rectangle(img, (gx, gy), (gx + bw, gy + bh), (shade, shade, shade), -1)
    cv2.line(img, (0, h // 2), (w, h // 2), (60, 60, 60), 3)
    cv2.line(img, (w // 2, 0), (w // 2, h), (60, 60, 60), 3)
    return img

city = make_city()
cv2.imwrite('data/city.png', city)
h0, w0 = city.shape[:2]

theta_tilt = np.deg2rad(45)
R_tilt = np.array([[np.cos(theta_tilt), -np.sin(theta_tilt)],
                    [np.sin(theta_tilt),  np.cos(theta_tilt)]])
tw = int(np.ceil(abs(w0 * R_tilt[0,0]) + abs(h0 * R_tilt[0,1])))
th = int(np.ceil(abs(w0 * R_tilt[1,0]) + abs(h0 * R_tilt[1,1])))
t_tilt = np.array([tw, th]) / 2 - R_tilt @ (np.array([w0, h0]) / 2)
M_tilt = np.hstack([R_tilt, t_tilt.reshape(2, 1)])
tilted = cv2.warpAffine(city, M_tilt, (tw, th), borderValue=(20, 20, 20))

plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(tilted, cv2.COLOR_BGR2RGB))
plt.title('Incoming photo (tilted 45°)')
plt.axis('off')
plt.savefig('output/t2_00_tilted_input.png', bbox_inches='tight', dpi=150)
plt.show()

h1, w1 = tilted.shape[:2]
theta = np.deg2rad(-45)

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
print('Rotation matrix R(-45°) =\n', R)

new_w = int(np.ceil(abs(w1 * R[0, 0]) + abs(h1 * R[0, 1])))
new_h = int(np.ceil(abs(w1 * R[1, 0]) + abs(h1 * R[1, 1])))
print(f'Expanded canvas: {new_w} x {new_h} (input was {w1} x {h1})')

old_center = np.array([w1 / 2, h1 / 2])
new_center = np.array([new_w / 2, new_h / 2])
t = new_center - R @ old_center

M = np.hstack([R, t.reshape(2, 1)])
print('Full 2x3 affine matrix M =\n', M)

corrected = cv2.warpAffine(tilted, M, (new_w, new_h), borderValue=(20, 20, 20))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(tilted, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Tilted input {w1}x{h1}')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Corrected, no corners clipped {new_w}x{new_h}')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('output/t2_01_corrected.png', bbox_inches='tight', dpi=150)
plt.show()