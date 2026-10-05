import argparse
import os
import cv2
import numpy as np


class SiftRecognizer:
    def __init__(self, ref_gray, min_inliers=10, ratio=0.75):
        self.sift = cv2.SIFT_create()
        self.kp, self.des = self.sift.detectAndCompute(ref_gray, None)
        h, w = ref_gray.shape
        self.box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        self.bf = cv2.BFMatcher()
        self.min_inliers, self.ratio = min_inliers, ratio

    def locate(self, frame_gray):
        kp, des = self.sift.detectAndCompute(frame_gray, None)
        if des is None or len(kp) < 4:
            return None, 0
        good = [m[0] for m in self.bf.knnMatch(self.des, des, k=2)
                if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
        if len(good) < self.min_inliers:
            return None, len(good)
        src = np.float32([self.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or int(inl.sum()) < self.min_inliers:
            return None, 0
        corners = cv2.perspectiveTransform(self.box, H)
        if not cv2.isContourConvex(corners.astype(np.int32)):
            return None, 0
        return corners, int(inl.sum())


def make_demo(folder):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(3)
    ref = cv2.resize(rng.integers(0, 255, (20, 28, 3), dtype=np.uint8), (280, 200), interpolation=cv2.INTER_CUBIC)
    cv2.putText(ref, "TARGET", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    cv2.imwrite(os.path.join(folder, "ref.png"), ref)

    bg = cv2.resize(rng.integers(80, 140, (24, 32, 3), dtype=np.uint8), (640, 480))
    vp = os.path.join(folder, "demo_video.mp4")
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
    h, w = ref.shape[:2]
    for t in range(100):
        frame = bg.copy()
        scale = 0.6 + 0.4 * np.sin(t / 15.0) ** 2
        angle = 25 * np.sin(t / 20.0)
        cx, cy = 150 + 4.5 * t, 240 + 80 * np.sin(t / 12.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += np.array([cx, cy]) - np.array([w / 2, h / 2])
        warped = cv2.warpAffine(ref, M, (640, 480))
        m = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (640, 480))
        frame[m > 0] = warped[m > 0]
        if 55 < t < 75:                                       
            cv2.rectangle(frame, (int(cx) - 20, int(cy) - 90), (int(cx) + 60, int(cy) + 10), (40, 40, 40), -1)
        vw.write(frame)
    vw.release()
    return os.path.join(folder, "ref.png"), vp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--video")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.demo or not (a.ref and a.video):
        a.ref, a.video = make_demo(a.out)
    ref = cv2.imread(a.ref, cv2.IMREAD_GRAYSCALE)
    rec = SiftRecognizer(ref)
    cap = cv2.VideoCapture(int(a.video) if a.video.isdigit() else a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    W, H = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(os.path.join(a.out, "task4_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = hit = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        corners, inl = rec.locate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if corners is not None:
            hit += 1
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            cv2.putText(frame, f"Object ({inl} inliers)", tuple(corners[0, 0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        vw.write(frame)
        if a.show:
            cv2.imshow("SIFT recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release(); vw.release(); cv2.destroyAllWindows()
    print(f"Frames: {n} | object found in {hit} ({100*hit/max(n,1):.0f}%)")


if __name__ == "__main__":
    main()
