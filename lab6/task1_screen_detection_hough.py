import argparse
import os
import cv2
import numpy as np


def make_demo_image(path):
    img = np.full((480, 800, 3), (200, 205, 210), np.uint8)
    sw, sh = 140, 90
    for r in range(2):
        for c in range(4):
            if r == 1 and c == 2:         
                continue
            x, y = 50 + c * 190, 60 + r * 200
            cv2.rectangle(img, (x - 10, y - 10), (x + sw + 10, y + sh + 10), (20, 20, 20), -1)
            on = (r + c) % 2 == 0
            cv2.rectangle(img, (x, y), (x + sw, y + sh), (230, 200, 90) if on else (75, 75, 75), -1)
            cv2.rectangle(img, (x + 50, y + sh + 10), (x + 90, y + sh + 30), (90, 90, 90), -1)
    cv2.imwrite(path, img)


def detect_screens(img, min_area=1500, max_area_frac=0.2, on_thresh=110):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=25)

    mask = np.zeros_like(gray)
    kept = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang < 10 or ang > 170 or abs(ang - 90) < 10:      
                L = max(np.hypot(x2 - x1, y2 - y1), 1)
                dx, dy = (x2 - x1) / L * 25, (y2 - y1) / L * 25     
                cv2.line(mask, (int(x1 - dx), int(y1 - dy)), (int(x2 + dx), int(y2 + dy)), 255, 3)
                kept.append((x1, y1, x2, y2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

    # regions enclosed by the detected lines
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), connectivity=4)
    H, W = gray.shape
    screens = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:         
            continue
        if a < min_area or a > max_area_frac * H * W:
            continue
        if a / float(w * h) < 0.75 or not (1.0 <= w / float(h) <= 2.5):
            continue
        b = float(gray[y + 3:y + h - 3, x + 3:x + w - 3].mean())
        screens.append({"box": (x, y, w, h), "brightness": b, "status": "ON" if b > on_thresh else "OFF"})
    return screens, kept


def find_missing(screens, tol=0.5):
    if not screens:
        return []
    cs = sorted(((s["box"][0] + s["box"][2] / 2, s["box"][1] + s["box"][3] / 2, s["box"][3]) for s in screens),
                key=lambda t: t[1])
    rows, med_h = [], np.median([c[2] for c in cs])
    for c in cs:
        if rows and abs(c[1] - np.mean([r[1] for r in rows[-1]])) < tol * med_h:
            rows[-1].append(c)
        else:
            rows.append([c])
    ref = max(rows, key=len)
    ref_x = sorted(c[0] for c in ref)
    tol_x = 0.5 * np.median(np.diff(ref_x)) if len(ref_x) > 1 else 50
    missing = []
    for ri, row in enumerate(rows):
        xs = [c[0] for c in row]
        for rx in ref_x:
            if not any(abs(rx - x) < tol_x for x in xs):
                missing.append((ri + 1, int(rx), int(np.mean([c[1] for c in row]))))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.demo or not a.image:
        a.image = os.path.join(a.out, "demo_lab.png")
        make_demo_image(a.image)
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit("Could not read image")

    screens, lines = detect_screens(img)
    missing = find_missing(screens)

    out = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for s in screens:
        x, y, w, h = s["box"]
        col = (0, 200, 0) if s["status"] == "ON" else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 3)
        cv2.putText(out, s["status"], (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    for row, cx, cy in missing:
        cv2.putText(out, "MISSING", (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.circle(out, (cx, cy), 8, (255, 0, 255), -1)

    print(f"Screens detected: {len(screens)} | ON: {sum(s['status']=='ON' for s in screens)} "
          f"| OFF: {sum(s['status']=='OFF' for s in screens)}")
    for row, cx, cy in missing:
        print(f"Anomaly: missing screen in row {row} near x={cx}")
    cv2.imwrite(os.path.join(a.out, "task1_result.png"), out)
    if a.show:
        cv2.imshow("Screens", out); cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()