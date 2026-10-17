import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_image(path=None):
    if path:
        img = cv2.imread(path)
        if img is not None:
            return img
    img = np.full((300, 400, 3), 255, np.uint8)
    for x in range(0, 400, 40):
        cv2.line(img, (x, 0), (x, 299), (200, 200, 200), 1)
    for y in range(0, 300, 40):
        cv2.line(img, (0, y), (399, y), (200, 200, 200), 1)
    cv2.rectangle(img, (60, 60), (340, 240), (0, 0, 255), 4)
    cv2.circle(img, (200, 150), 60, (255, 0, 0), 4)
    cv2.line(img, (60, 60), (340, 240), (0, 160, 0), 4)
    cv2.putText(img, "CV", (160, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


def translate(img, tx, ty):
    h, w = img.shape[:2]
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def rotate(img, angle, scale=1.0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def scale_img(img, sx, sy):
    h, w = img.shape[:2]
    m = np.float32([[sx, 0, 0], [0, sy, 0]])
    return cv2.warpAffine(img, m, (int(w * sx), int(h * sy)), borderValue=(255, 255, 255))


def shear(img, shx, shy):
    h, w = img.shape[:2]
    m = np.float32([[1, shx, 0], [shy, 1, 0]])
    return cv2.warpAffine(img, m, (int(w + abs(shx) * h), int(h + abs(shy) * w)), borderValue=(255, 255, 255))


def show_grid(images, titles, cols=3, suptitle=None, save=None):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120)
    plt.show()


def task_a(img):
    results = [
        img,
        translate(img, 80, 50),
        rotate(img, 30),
        scale_img(img, 0.6, 0.6),
        scale_img(img, 1.3, 0.8),
        shear(img, 0.4, 0.0),
    ]
    titles = ["Original", "Translation (80, 50)", "Rotation 30 deg", "Scaling 0.6x", "Non-uniform scaling (1.3, 0.8)", "Shearing (0.4, 0)"]
    show_grid(results, titles, suptitle="Q1(a): 2D transformations", save="q1a_2d_transformations.png")


def rotation_matrix_3d(ax_deg, ay_deg, az_deg):
    ax, ay, az = np.radians([ax_deg, ay_deg, az_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def cube_vertices():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    return v - 0.5


def cube_faces(v):
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [[v[i] for i in f] for f in idx]


def draw_cube(ax, v, color, title):
    ax.add_collection3d(Poly3DCollection(cube_faces(v), facecolors=color, edgecolors="k", alpha=0.6))
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_zlim(-2, 3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)


def task_b():
    v = cube_vertices()
    rotated = (rotation_matrix_3d(30, 45, 20) @ v.T).T
    moved = v + np.array([1.5, 1.0, 1.0])
    both = (rotation_matrix_3d(30, 45, 20) @ v.T).T + np.array([1.5, 1.0, 1.0])
    fig = plt.figure(figsize=(16, 4.5))
    for i, (vs, c, t) in enumerate([(v, "cyan", "Original"), (rotated, "orange", "Rotated (30, 45, 20)"), (moved, "lightgreen", "Translated (1.5, 1, 1)"), (both, "violet", "Rotated + Translated")]):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        draw_cube(ax, vs, c, t)
    plt.tight_layout()
    plt.savefig("q1b_3d_transformation.png", dpi=120)
    plt.show()


def task_c():
    x, y = 3.0, 4.0
    p_h = np.array([x, y, 1.0])
    print("2D point (x, y):", (x, y))
    print("Homogeneous coordinates (x, y, 1):", p_h)
    t = np.array([[1, 0, 5], [0, 1, -2], [0, 0, 1]], dtype=float)
    theta = np.radians(90)
    r = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    combined = t @ r @ s
    print("Translation matrix:\n", t)
    print("Rotation matrix:\n", np.round(r, 3))
    print("Scaling matrix:\n", s)
    print("Combined T @ R @ S:\n", np.round(combined, 3))
    out = combined @ p_h
    print("Transformed homogeneous point:", np.round(out, 3))
    print("Back to Cartesian:", np.round(out[:2] / out[2], 3))
    scaled = 5 * p_h
    print("Scaled homogeneous (5x, 5y, 5):", scaled, "-> Cartesian:", scaled[:2] / scaled[2])
    print()
    print("Why the extra coordinate is used in computer vision:")
    print("1. Translation is not linear in 2D Cartesian coordinates, but becomes a matrix multiplication in homogeneous coordinates.")
    print("2. Rotation, scaling, shearing and translation can all be chained into a single 3x3 matrix.")
    print("3. Projective transformations (perspective division by w) can be expressed uniformly.")
    print("4. Points at infinity (w = 0) and vanishing points can be represented.")
    print("5. The point (x, y, 1) is equivalent to (kx, ky, k) for any k != 0.")


def task_d(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[40, 60], [w - 80, 20], [80, h - 40]])
    m = cv2.getAffineTransform(src, dst)
    out = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    print("Affine matrix:\n", m)
    pts = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=float).T
    mapped = (np.vstack([m, [0, 0, 1]]) @ pts)[:2].T
    e1, e2 = pts[:2, 1] - pts[:2, 0], pts[:2, 3] - pts[:2, 0]
    f1, f2 = mapped[1] - mapped[0], mapped[3] - mapped[0]
    cross_before = e1[0] * e2[1] - e1[1] * e2[0]
    cross_after = f1[0] * f2[1] - f1[1] * f2[0]
    print("Mapped corners:\n", np.round(mapped, 2))
    print("Parallelogram check (opposite edges equal):", np.allclose(mapped[1] - mapped[0], mapped[2] - mapped[3]))
    print("Area ratio (det of linear part):", round(cross_after / cross_before, 4))
    print("Straight lines remain straight: True")
    print("Parallel lines remain parallel: True")
    print("Angles and lengths are not preserved in general.")
    marked = img.copy()
    cv2.line(marked, (0, h // 2), (w - 1, h // 2), (0, 0, 255), 3)
    cv2.line(marked, (0, h // 4), (w - 1, h // 4), (0, 0, 255), 3)
    cv2.line(marked, (w // 2, 0), (w // 2, h - 1), (255, 0, 0), 3)
    out_marked = cv2.warpAffine(marked, m, (w, h), borderValue=(255, 255, 255))
    show_grid([marked, out_marked], ["Original with lines", "After affine (lines stay straight and parallel)"], cols=2, suptitle="Q1(d): Affine transformation", save="q1d_affine.png")


def make_tilted_book(w=600, h=800):
    page = np.full((h, w, 3), 240, np.uint8)
    cv2.rectangle(page, (10, 10), (w - 10, h - 10), (30, 30, 150), 8)
    cv2.putText(page, "BOOK", (170, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
    for i in range(10):
        y = 230 + i * 45
        cv2.line(page, (60, y), (w - 60, y), (60, 60, 60), 3)
    cv2.circle(page, (w // 2, 640), 60, (0, 120, 0), 6)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[180, 120], [620, 60], [700, 560], [90, 520]])
    m = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((650, 800, 3), 200, np.uint8)
    tilted = cv2.warpPerspective(page, m, (800, 650), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    return tilted, dst


def task_e(path=None):
    if path:
        tilted = cv2.imread(path)
        pts = None
    else:
        tilted, pts = make_tilted_book()
    if pts is None:
        pts = np.float32([[180, 120], [620, 60], [700, 560], [90, 520]])
    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bot = np.linalg.norm(pts[2] - pts[3])
    h_l = np.linalg.norm(pts[3] - pts[0])
    h_r = np.linalg.norm(pts[2] - pts[1])
    out_w, out_h = int(max(w_top, w_bot)), int(max(h_l, h_r))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    hmat = cv2.getPerspectiveTransform(np.float32(pts), dst)
    print("Homography matrix:\n", np.round(hmat, 5))
    flat = cv2.warpPerspective(tilted, hmat, (out_w, out_h))
    marked = tilted.copy()
    cv2.polylines(marked, [np.int32(pts)], True, (0, 0, 255), 3)
    show_grid([marked, flat], ["Tilted book (corners marked)", "Corrected (flat)"], cols=2, suptitle="Q1(e): Homography", save="q1e_homography.png")
    cv2.imwrite("q1e_corrected.png", flat)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    img = load_image(path)
    task_a(img)
    task_b()
    task_c()
    task_d(img)
    task_e()


if __name__ == "__main__":
    main()