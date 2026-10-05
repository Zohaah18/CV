import argparse
import glob
import os
import cv2
import numpy as np


def stitch_pair(base, new, sift, bf, ratio=0.75):
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), None)
    good = [m[0] for m in bf.knnMatch(d2, d1, k=2) if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 10:
        raise RuntimeError(f"Not enough matches ({len(good)})")
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)

    h1, w1 = base.shape[:2]
    h2, w2 = new.shape[:2]
    c_new = cv2.perspectiveTransform(np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2), H)
    c_base = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    allc = np.concatenate([c_base, c_new])
    xmin, ymin = np.floor(allc.min(axis=(0, 1))).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=(0, 1))).astype(int)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], float)

    warped = cv2.warpPerspective(new, T @ H, (xmax - xmin, ymax - ymin))
    canvas = np.zeros_like(warped)
    canvas[-ymin:-ymin + h1, -xmin:-xmin + w1] = base
    mb, mw = canvas.any(2), warped.any(2)
    out = np.where((mw & ~mb)[..., None], warped, canvas)
    both = mb & mw
    out[both] = ((canvas[both].astype(np.int16) + warped[both].astype(np.int16)) // 2).astype(np.uint8)
    return out, len(good)


def crop_black(img):
    ys, xs = np.where(img.any(2))
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def make_demo(folder):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(5)
    wide = cv2.resize(rng.integers(0, 255, (30, 120, 3), dtype=np.uint8), (1300, 400), interpolation=cv2.INTER_CUBIC)
    for _ in range(60):
        cv2.circle(wide, (int(rng.integers(0, 1300)), int(rng.integers(0, 400))), int(rng.integers(8, 30)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    for i, x in enumerate(range(0, 1300, 220)):
        cv2.putText(wide, str(i), (x + 60, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    paths = []
    for i, x in enumerate((0, 350, 700)):
        crop = wide[:, x:x + 600].astype(np.float32) * (0.92 + 0.05 * i)     
        p = os.path.join(folder, f"part{i}.png")
        cv2.imwrite(p, np.clip(crop, 0, 255).astype(np.uint8))
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--compare", action="store_true", help="also run OpenCV's built-in Stitcher")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.demo or not a.images:
        paths = make_demo(os.path.join(a.out, "demo_parts"))
    elif len(a.images) == 1 and os.path.isdir(a.images[0]):
        paths = sorted(glob.glob(os.path.join(a.images[0], "*.*")))
    else:
        paths = a.images
    imgs = [cv2.imread(p) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if len(imgs) < 2:
        raise SystemExit("Need at least 2 images")

    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()
    pano = imgs[0]
    for i, im in enumerate(imgs[1:], 2):
        pano, n = stitch_pair(pano, im, sift, bf)
        print(f"Stitched image {i}/{len(imgs)} using {n} good matches -> {pano.shape[1]}x{pano.shape[0]}")
    pano = crop_black(pano)
    cv2.imwrite(os.path.join(a.out, "task5_panorama.png"), pano)

    if a.compare:
        st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, ref = st.stitch(imgs)
        if status == cv2.Stitcher_OK:
            cv2.imwrite(os.path.join(a.out, "task5_builtin.png"), ref)
            print("Built-in stitcher result saved")
    if a.show:
        cv2.imshow("Panorama", pano); cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
