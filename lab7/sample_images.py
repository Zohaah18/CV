import os
import sys
import cv2
import numpy as np

W, H = 960, 720
FOCAL = 850.0
TEX = 512


def make_texture(seed):
    rng = np.random.default_rng(seed)
    tex = rng.integers(0, 255, (TEX // 8, TEX // 8, 3), dtype=np.uint8)
    tex = cv2.resize(tex, (TEX, TEX), interpolation=cv2.INTER_CUBIC)
    fine = rng.integers(0, 255, (TEX // 2, TEX // 2, 3), dtype=np.uint8)
    fine = cv2.resize(fine, (TEX, TEX), interpolation=cv2.INTER_LINEAR)
    tex = cv2.addWeighted(tex, 0.6, fine, 0.4, 0)
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, TEX)), int(rng.integers(0, TEX)))
        if rng.random() < 0.5:
            cv2.circle(tex, p, int(rng.integers(6, 28)), c, -1)
        else:
            q = (int(rng.integers(0, TEX)), int(rng.integers(0, TEX)))
            cv2.line(tex, p, q, c, int(rng.integers(2, 6)))
    for _ in range(25):
        p = (int(rng.integers(0, TEX - 60)), int(rng.integers(20, TEX)))
        cv2.putText(tex, str(int(rng.integers(0, 999))), p, cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 1.4)), (0, 0, 0), 2)
    return tex


def camera_pose(theta, radius, height):
    c = np.array([radius * np.cos(theta), radius * np.sin(theta), height])
    f = -c / np.linalg.norm(c)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    r = np.vstack([right, down, f])
    t = -r @ c
    return r, t, c


def project(points, r, t):
    k = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]])
    cam = (r @ points.T).T + t
    if np.any(cam[:, 2] < 0.2):
        return None
    uv = (k @ cam.T).T
    return (uv[:, :2] / uv[:, 2:3]).astype(np.float32), cam[:, 2]


def draw_quad(canvas, tex, quad3d, r, t):
    res = project(quad3d, r, t)
    if res is None:
        return
    uv, _ = res
    src = np.float32([[0, 0], [TEX - 1, 0], [TEX - 1, TEX - 1], [0, TEX - 1]])
    m = cv2.getPerspectiveTransform(src, uv)
    warped = cv2.warpPerspective(tex, m, (W, H), flags=cv2.INTER_LINEAR)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(uv)), 255)
    canvas[mask > 0] = warped[mask > 0]


def cube_faces(s=0.5):
    v = np.array([[-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
                  [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]], dtype=float)
    idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]
    return [v[i] for i in idx]


def render(theta, radius, height, textures, ground_tex):
    r, t, c = camera_pose(theta, radius, height)
    canvas = np.full((H, W, 3), (200, 190, 180), np.uint8)
    g = 1.8
    ground = np.array([[-g, -g, -0.5], [g, -g, -0.5], [g, g, -0.5], [-g, g, -0.5]])
    draw_quad(canvas, ground_tex, ground, r, t)
    faces = cube_faces()
    order = sorted(range(6), key=lambda i: -np.linalg.norm(faces[i].mean(axis=0) - c))
    for i in order:
        n = np.cross(faces[i][1] - faces[i][0], faces[i][3] - faces[i][0])
        if np.dot(n, faces[i].mean(axis=0) - c) > 0:
            n = -n
        if np.dot(n, c - faces[i].mean(axis=0)) <= 0:
            continue
        draw_quad(canvas, textures[i], faces[i], r, t)
    return canvas


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "images"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 36
    os.makedirs(out, exist_ok=True)
    textures = [make_texture(100 + i) for i in range(6)]
    ground_tex = make_texture(999)
    for i in range(count):
        theta = 2 * np.pi * i / count
        ring = i % 3
        radius = 3.2 + 0.3 * ring
        height = [1.6, 2.4, 3.2][ring]
        img = render(theta, radius, height, textures, ground_tex)
        cv2.imwrite(os.path.join(out, f"view_{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Wrote {count} images to {out}")


if __name__ == "__main__":
    main()