import argparse
import glob
import os
import cv2
import numpy as np


def roi_mask(edges):
    h, w = edges.shape
    poly = np.array([[(int(0.05 * w), h), (int(0.45 * w), int(0.62 * h)),
                      (int(0.55 * w), int(0.62 * h)), (int(0.95 * w), h)]], np.int32)
    m = np.zeros_like(edges)
    cv2.fillPoly(m, poly, 255)
    return cv2.bitwise_and(edges, m)


def average_lane(segments, h):
    left, right = [], []
    for x1, y1, x2, y2 in segments:
        if x1 == x2:
            continue
        slope, b = np.polyfit((x1, x2), (y1, y2), 1)
        if abs(slope) < 0.4:                     
            continue
        (left if slope < 0 else right).append((slope, b))
    lanes = []
    for side in (left, right):
        if side:
            s, b = np.mean(side, axis=0)
            y1, y2 = h, int(0.62 * h)
            lanes.append((int((y1 - b) / s), y1, int((y2 - b) / s), y2))
    return lanes


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    roi = roi_mask(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=100)
    segs = [] if lines is None else [tuple(int(v) for v in l) for l in lines.reshape(-1, 4)]
    lanes = average_lane(segs, frame.shape[0])
    out = frame.copy()
    overlay = np.zeros_like(frame)
    for x1, y1, x2, y2 in lanes:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 10)
    if len(lanes) == 2:                                         
        pts = np.array([[lanes[0][0], lanes[0][1]], [lanes[0][2], lanes[0][3]],
                        [lanes[1][2], lanes[1][3]], [lanes[1][0], lanes[1][1]]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 120, 0))
    return cv2.addWeighted(out, 1.0, overlay, 0.6, 0), len(lanes)


def make_demo(path):
    h, w = 540, 960
    img = np.full((h, w, 3), (90, 140, 90), np.uint8)
    img[: int(0.5 * h)] = (235, 206, 135)                        
    vp = (w // 2, int(0.5 * h))
    cv2.fillPoly(img, [np.array([(0, h), (w, h), (vp[0] + 20, vp[1]), (vp[0] - 20, vp[1])])], (70, 70, 70))
    cv2.line(img, (140, h), (vp[0] - 10, vp[1]), (255, 255, 255), 10)
    cv2.line(img, (w - 140, h), (vp[0] + 10, vp[1]), (0, 220, 255), 10)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--folder")
    ap.add_argument("--video")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.video:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        vw = cv2.VideoWriter(os.path.join(a.out, "task6_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (int(cap.get(3)), int(cap.get(4))))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            res, _ = detect_lanes(f)
            vw.write(res)
            if a.show:
                cv2.imshow("Lanes", res)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release(); vw.release(); cv2.destroyAllWindows()
        print("Saved", os.path.join(a.out, "task6_result.mp4"))
        return

    if a.folder:
        paths = sorted(glob.glob(os.path.join(a.folder, "*.*")))
    elif a.images:
        paths = a.images
    else:
        p = os.path.join(a.out, "demo_road.png"); make_demo(p); paths = [p]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        res, n = detect_lanes(img)
        name = os.path.join(a.out, "task6_" + os.path.basename(p))
        cv2.imwrite(name, res)
        print(f"{os.path.basename(p)}: {n} lane line(s) detected -> {name}")
        if a.show:
            cv2.imshow("Lanes", res); cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()