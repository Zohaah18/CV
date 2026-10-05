import argparse
import os
import cv2
import numpy as np


def detect_coins(img, min_r=20, max_r=80, param1=100, param2=40, min_dist=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist or min_r * 1.6,
                               param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r)
    return [] if circles is None else np.round(circles.reshape(-1, 3)).astype(int).tolist()


def make_demo(path):
    rng = np.random.default_rng(11)
    img = np.full((500, 700, 3), (60, 90, 120), np.uint8)
    img = cv2.add(img, rng.integers(0, 15, img.shape, dtype=np.uint8))
    spots = [(120, 120, 45), (260, 150, 55), (420, 110, 40), (570, 160, 60),
             (150, 330, 60), (320, 360, 42), (470, 330, 52), (600, 400, 38)]
    for x, y, r in spots:
        cv2.circle(img, (x, y), r, (150, 190, 215), -1)
        cv2.circle(img, (x, y), r, (90, 130, 160), 3)
        cv2.circle(img, (x, y), int(r * 0.7), (130, 170, 200), 2)
    cv2.imwrite(path, cv2.GaussianBlur(img, (3, 3), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--min-r", type=int, default=20)
    ap.add_argument("--max-r", type=int, default=80)
    ap.add_argument("--param2", type=int, default=40)
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = a.images
    if a.demo or not paths:
        p = os.path.join(a.out, "demo_coins.png"); make_demo(p); paths = [p]

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        coins = detect_coins(img, a.min_r, a.max_r, param2=a.param2)
        coins.sort(key=lambda c: (c[1] // 100, c[0]))
        out = img.copy()
        for i, (x, y, r) in enumerate(coins, 1):
            cv2.circle(out, (x, y), r, (0, 255, 0), 3)
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(out, str(i), (x - 10, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, f"Coins: {len(coins)}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        name = os.path.join(a.out, "task7_" + os.path.basename(p))
        cv2.imwrite(name, out)
        print(f"{os.path.basename(p)}: {len(coins)} coins | radii: {[c[2] for c in coins]}")
        if a.show:
            cv2.imshow("Coins", out); cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()