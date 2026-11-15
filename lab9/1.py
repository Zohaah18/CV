import argparse
import collections
import time

import cv2
from ultralytics import YOLO

LAB_ASSET_CLASSES = {
    "laptop", "mouse", "keyboard", "cell phone", "tv", "chair", "bottle",
    "book", "cup", "person", "remote", "backpack", "clock", "scissors",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="any YOLO weights (auto-downloaded)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--assets", nargs="*", default=None,
                    help="only show these class names (default: all COCO classes)")
    args = ap.parse_args()

    model = YOLO(args.model)
    wanted = set(args.assets) if args.assets else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1 or close other apps using it.")

    max_seen = collections.Counter()   
    fps, prev = 0.0, time.time()
    snap = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        names = res.names
        frame_count = collections.Counter()

        for box in res.boxes:
            cls_name = names[int(box.cls[0])]
            if wanted and cls_name not in wanted:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = (0, 200, 0) if cls_name in LAB_ASSET_CLASSES else (255, 160, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)
            frame_count[cls_name] += 1

        for k, v in frame_count.items():
            max_seen[k] = max(max_seen[k], v)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  objects: {sum(frame_count.values())}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Q1 - YOLO lab asset detection (q to quit, s to save)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snap += 1
            cv2.imwrite(f"lab_assets_snapshot_{snap}.jpg", frame)
            print(f"saved lab_assets_snapshot_{snap}.jpg")

    cap.release()
    cv2.destroyAllWindows()

    print("\nLab assets detected in this session (max seen at once):")
    for k, v in max_seen.most_common():
        print(f"  {k:15s} x{v}")


if __name__ == "__main__":
    main()
