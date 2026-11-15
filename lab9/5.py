import argparse
import csv
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
            "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
ANGLES = [("L elbow", 5, 7, 9), ("R elbow", 6, 8, 10),
          ("L knee", 11, 13, 15), ("R knee", 12, 14, 16),
          ("L shoulder", 7, 5, 11), ("R shoulder", 8, 6, 12),
          ("L hip", 5, 11, 13), ("R hip", 6, 12, 14)]
KP_CONF = 0.5

IMG_DIR = Path("sample_images")
PERSON_URLS = ["https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
               "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
               "https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg"] + \
              [f"http://images.cocodataset.org/val2017/{i:012d}.jpg" for i in (785, 872, 885, 1000, 1268)]


def joint_angle(a, b, c):
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def random_person_image(seed=None):
    rng = random.Random(seed)
    IMG_DIR.mkdir(exist_ok=True)
    urls = PERSON_URLS[:]
    rng.shuffle(urls)
    for url in urls:
        dest = IMG_DIR / url.split("/")[-1]
        if not dest.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                dest.write_bytes(urllib.request.urlopen(req, timeout=20).read())
            except Exception as e:
                print(f"(download failed {dest.name}: {type(e).__name__})")
                continue
        img = cv2.imread(str(dest))
        if img is not None:
            print(f"Using random picture: {dest.name}")
            return img
    cached = [cv2.imread(str(p)) for p in IMG_DIR.glob("*.jp*g")]
    cached = [c for c in cached if c is not None]
    if cached:
        return rng.choice(cached)
    raise SystemExit("No picture available offline - put a photo of a person in ./sample_images")


class Smoother:

    def __init__(self, alpha=0.6):
        self.alpha, self.prev = alpha, None

    def __call__(self, xy):
        self.prev = xy.copy() if self.prev is None else self.alpha * xy + (1 - self.alpha) * self.prev
        return self.prev

    def reset(self):
        self.prev = None


def draw_person(img, xy, conf):
    for a, b in SKELETON:
        if conf[a] > KP_CONF and conf[b] > KP_CONF:
            col = (0, 165, 255) if (a in LEFT or b in LEFT) and not (a in RIGHT or b in RIGHT) else \
                  (255, 120, 0) if (a in RIGHT or b in RIGHT) and not (a in LEFT or b in LEFT) else (0, 220, 0)
            cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), col, 3, cv2.LINE_AA)
    for i, (p, c) in enumerate(zip(xy, conf)):
        if c > KP_CONF:
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_angles(img, xy, conf):
    vals = {}
    for name, a, b, c in ANGLES:
        if min(conf[a], conf[b], conf[c]) > KP_CONF:
            ang = joint_angle(xy[a], xy[b], xy[c])
            vals[name] = ang
            x, y = int(xy[b][0]) + 8, int(xy[b][1]) - 8
            cv2.putText(img, f"{ang:.0f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, f"{ang:.0f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vals


def draw_panel(img, xy, conf, angles, fps, n_people):
    lines = [f"FPS {fps:4.1f}   persons: {n_people}", "-- keypoints (x, y, conf) --"]
    for name, p, c in zip(KP_NAMES, xy, conf):
        lines.append(f"{name:11s}{p[0]:6.0f}{p[1]:6.0f}  {c:.2f}" if c > KP_CONF else f"{name:11s}   --     --   --")
    if angles:
        lines.append("-- joint angles --")
        lines += [f"{k:11s}{v:6.0f} deg" for k, v in angles.items()]
    h, w = img.shape[:2]
    lh, pw = 17, 245
    ph = min(h - 10, lh * len(lines) + 12)
    ov = img.copy()
    cv2.rectangle(ov, (5, 5), (5 + pw, 5 + ph), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for i, t in enumerate(lines):
        if 5 + 12 + i * lh > 5 + ph:
            break
        cv2.putText(img, t, (10, 22 + i * lh), cv2.FONT_HERSHEY_PLAIN, 1.05, (255, 255, 255), 1, cv2.LINE_AA)


def process(frame, model, smoother, show_angles, show_panel, fps, imgsz, conf_thr, device):
    r = model(frame, imgsz=imgsz, conf=conf_thr, device=device, verbose=False)[0]
    out = frame.copy()
    primary = None
    n = 0
    if r.keypoints is not None and len(r.keypoints.xy):
        xys = r.keypoints.xy.cpu().numpy()                   
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xys.shape[:2])
        areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
        main_i = int(areas.argmax())                         
        n = len(xys)
        for i in range(n):
            if i == main_i:
                continue
            draw_person(out, xys[i], confs[i])
        xy_s = smoother(xys[main_i])
        draw_person(out, xy_s, confs[main_i])
        angles = draw_angles(out, xy_s, confs[main_i]) if show_angles else {}
        primary = (xy_s, confs[main_i], angles)
        if show_panel:
            draw_panel(out, xy_s, confs[main_i], angles, fps, n)
    else:
        smoother.reset()
        cv2.putText(out, "no person detected", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out, primary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "image", "video"], default="webcam")
    ap.add_argument("--video", default=None)
    ap.add_argument("--image", default=None, help="your own image (with --source image)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="yolov8n-pose.pt",
                    help="yolov8n-pose.pt (fastest) ... yolov8s/m-pose.pt (more accurate)")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-angles", action="store_true")
    ap.add_argument("--log", default=None, help="CSV file to log keypoint coordinates per frame")
    ap.add_argument("--print-coords", action="store_true", help="print coordinates to console")
    ap.add_argument("--save-video", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    smoother = Smoother()
    show_angles, show_panel = not args.no_angles, True

    log_f = log_w = None
    if args.log:
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["time_s"] + [f"{n}_{ax}" for n in KP_NAMES for ax in ("x", "y")] +
                       [f"angle_{a[0].replace(' ', '_')}" for a in ANGLES])
    if args.source == "image":
        img = cv2.imread(args.image) if args.image else random_person_image()
        out, prim = process(img, model, smoother, show_angles, True, 0.0, args.imgsz, args.conf, args.device)
        cv2.imwrite("pose_result.jpg", out)
        print("Saved pose_result.jpg")
        if prim:
            for n, p, c in zip(KP_NAMES, prim[0], prim[1]):
                print(f"{n:11s} x={p[0]:7.1f} y={p[1]:7.1f} conf={c:.2f}")
            for k, v in prim[2].items():
                print(f"{k:11s} {v:6.1f} deg")
        if sys.platform != "linux" or os.environ.get("DISPLAY"):   
            cv2.imshow("Q5 - Pose estimation", out)
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(args.video if args.source == "video" else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the video source (try --camera 1, or --source image).")
    writer = None
    fps, prev, t_start, snap = 0.0, time.time(), time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.source == "webcam":
            frame = cv2.flip(frame, 1)                  
        out, prim = process(frame, model, smoother, show_angles, show_panel, fps, args.imgsz, args.conf, args.device)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
        prev = now

        if prim:
            xy, conf, angles = prim
            if args.print_coords:
                print(f"t={now - t_start:6.2f}s " + " ".join(
                    f"{n}=({p[0]:.0f},{p[1]:.0f})" for n, p, c in zip(KP_NAMES, xy, conf) if c > KP_CONF))
            if log_w:
                row = [f"{now - t_start:.3f}"]
                for p, c in zip(xy, conf):
                    row += [f"{p[0]:.1f}", f"{p[1]:.1f}"] if c > KP_CONF else ["", ""]
                row += [f"{angles[a[0]]:.1f}" if a[0] in angles else "" for a in ANGLES]
                log_w.writerow(row)

        if args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)

        cv2.imshow("Q5 - Real-time pose estimation (q quit | a angles | c panel | s save)", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("a"):
            show_angles = not show_angles
        elif k == ord("c"):
            show_panel = not show_panel
        elif k == ord("s"):
            snap += 1
            cv2.imwrite(f"pose_snapshot_{snap}.jpg", out)
            print(f"saved pose_snapshot_{snap}.jpg")

    cap.release()
    if writer:
        writer.release()
    if log_f:
        log_f.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
