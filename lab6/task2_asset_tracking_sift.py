import argparse
import csv
import glob
import os
import cv2
import numpy as np

COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def find_instances(tmpl_gray, scene_gray, sift, max_inst=6, min_inliers=12, ratio=0.75):
    kp_t, des_t = sift.detectAndCompute(tmpl_gray, None)
    if des_t is None:
        return []
    h, w = tmpl_gray.shape
    box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    mask = np.full(scene_gray.shape, 255, np.uint8)
    bf = cv2.BFMatcher()
    found = []
    for _ in range(max_inst):
        kp_s, des_s = sift.detectAndCompute(scene_gray, mask)
        if des_s is None or len(kp_s) < 4:
            break
        good = [m[0] for m in bf.knnMatch(des_t, des_s, k=2)
                if len(m) == 2 and m[0].distance < ratio * m[1].distance]
        if len(good) < min_inliers:
            break
        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or int(inl.sum()) < min_inliers:
            break
        corners = cv2.perspectiveTransform(box, H)
        if not cv2.isContourConvex(corners.astype(np.int32)):
            break
        found.append((corners, int(inl.sum())))
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 0)   
    return found


def texture(h, w, label, seed):
    rng = np.random.default_rng(seed)
    t = cv2.resize(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(15):
        cv2.circle(t, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(5, 20)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    cv2.putText(t, label, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return t


def paste(scene, tmpl, center, angle, scale):
    h, w = tmpl.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += np.array(center) - np.array([w / 2, h / 2])
    warped = cv2.warpAffine(tmpl, M, scene.shape[1::-1])
    m = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, scene.shape[1::-1])
    scene[m > 0] = warped[m > 0]


def make_demo(folder):
    os.makedirs(folder, exist_ok=True)
    tm = texture(160, 220, "MONITOR", 1)
    tk = texture(90, 260, "KEYBOARD", 2)
    cv2.imwrite(os.path.join(folder, "monitor.png"), tm)
    cv2.imwrite(os.path.join(folder, "keyboard.png"), tk)
    rng = np.random.default_rng(7)
    scene = cv2.resize(rng.integers(90, 160, (40, 60, 3), dtype=np.uint8), (900, 600))
    paste(scene, tm, (200, 150), 0, 1.0)
    paste(scene, tm, (650, 170), 15, 0.8)
    paste(scene, tk, (250, 450), -10, 1.1)
    scene = cv2.GaussianBlur(scene, (3, 3), 0)
    path = os.path.join(folder, "demo_scene.png")
    cv2.imwrite(path, scene)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--templates")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.demo or not (a.scene and a.templates):
        a.templates = os.path.join(a.out, "demo_assets")
        a.scene = make_demo(a.templates)
    scene = cv2.imread(a.scene)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    out, inventory = scene.copy(), []

    files = [f for f in sorted(glob.glob(os.path.join(a.templates, "*.*")))
             if os.path.basename(f) != os.path.basename(a.scene)]
    for i, f in enumerate(files):
        t = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if t is None:
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        inst = find_instances(t, gray, sift)
        inventory.append((name, len(inst)))
        for k, (corners, n) in enumerate(inst, 1):
            col = COLORS[i % len(COLORS)]
            cv2.polylines(out, [corners.astype(np.int32)], True, col, 3)
            p = corners[0, 0].astype(int)
            cv2.putText(out, f"{name} #{k} ({n} inl)", (p[0], max(p[1] - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    print("INVENTORY")
    for name, n in inventory:
        print(f"  {name:<15} {n}")
    with open(os.path.join(a.out, "task2_inventory.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["asset", "count"]); w.writerows(inventory)
    cv2.imwrite(os.path.join(a.out, "task2_result.png"), out)
    if a.show:
        cv2.imshow("Assets", out); cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
