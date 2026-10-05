import argparse
import os
import cv2
import numpy as np


def make_demo(path):
    rng = np.random.default_rng(2)
    bg = cv2.resize(rng.integers(100, 130, (12, 16, 3), dtype=np.uint8), (640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(bg, (30, 30), (150, 120), (160, 160, 160), -1)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
    for t in range(140):
        f = bg.copy()
        if t >= 25:
            x = -60 + (t - 25) * 6
            cv2.circle(f, (x, 260), 35, (30, 60, 220), -1)
            cv2.circle(f, (x, 260), 35, (10, 10, 10), 2)
        vw.write(f)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--zone", help="x1,y1,x2,y2")
    ap.add_argument("--min-area", type=int, default=800)
    ap.add_argument("--min-overlap", type=int, default=200, help="min overlapping pixels to trigger alarm")
    ap.add_argument("--warmup", type=int, default=20, help="frames used to learn the background")
    ap.add_argument("--beep", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.demo or not a.video:
        a.video = os.path.join(a.out, "demo_security.mp4"); make_demo(a.video)

    cap = cv2.VideoCapture(int(a.video) if a.video.isdigit() else a.video)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    zx1, zy1, zx2, zy2 = map(int, a.zone.split(",")) if a.zone else (int(.5 * W), int(.3 * H), int(.88 * W), int(.75 * H))
    zone_mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(zone_mask, (zx1, zy1), (zx2, zy2), 255, -1)

    mog = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vw = cv2.VideoWriter(os.path.join(a.out, "task8_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    alarm_frames, was_alarm, n = [], False, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        fg = mog.apply(frame, learningRate=0.05 if n <= a.warmup else 0.0005)
        if n <= a.warmup:
            vw.write(frame); continue
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        alarm = False
        for c in cnts:
            if cv2.contourArea(c) < a.min_area:
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            inside = cv2.countNonZero(cv2.bitwise_and(m, zone_mask)) >= a.min_overlap
            alarm |= inside
            col = (0, 0, 255) if inside else (0, 255, 0)
            cv2.drawContours(frame, [c], -1, col, 2)             
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(frame, "INTRUDER" if inside else "object", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 255) if alarm else (255, 200, 0), -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255) if alarm else (255, 200, 0), 2)
        if alarm:
            cv2.putText(frame, "ALARM: UNAUTHORIZED OBJECT IN ZONE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            alarm_frames.append(n)
            if not was_alarm:
                print(f"[ALARM] frame {n} ({n / fps:.1f}s): object entered security zone")
                if a.beep:
                    try:
                        import winsound; winsound.Beep(1500, 400)
                    except ImportError:
                        print("\a", end="")
        was_alarm = alarm
        vw.write(frame)
        if a.show:
            cv2.imshow("Security", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release(); vw.release(); cv2.destroyAllWindows()
    print(f"Frames: {n} | alarm frames: {len(alarm_frames)}"
          + (f" (first {alarm_frames[0]}, last {alarm_frames[-1]})" if alarm_frames else ""))


if __name__ == "__main__":
    main()
