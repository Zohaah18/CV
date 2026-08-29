"""Task 3: real-time enhancement and side-by-side monitoring of an echo video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA, OUTPUT = ROOT / "data", ROOT / "output"


def first_video(folder: Path) -> Path:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".mp4")
    if not files:
        raise FileNotFoundError(f"No .mp4 found in {folder}; add a short clip or pass --video.")
    return files[0]


def color_balance(bgr: np.ndarray) -> np.ndarray:
    values = bgr.astype(np.float32)
    means = values.mean(axis=(0, 1))
    return np.uint8(np.clip(values * (means.mean() / np.maximum(means, 1e-6)), 0, 255))


def enhance(frame: np.ndarray, gamma: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    logged = np.uint8(np.clip(255 / np.log1p(255) * np.log1p(equalized.astype(np.float32)), 0, 255))
    corrected = np.uint8(np.clip(255 * (logged.astype(np.float32) / 255) ** gamma, 0, 255))
    heatmap = cv2.applyColorMap(corrected, cv2.COLORMAP_JET)
    return color_balance(heatmap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Path to ultrasound MP4")
    parser.add_argument("--gamma", type=float, default=0.6, help="Fractional power-law exponent")
    parser.add_argument("--no-display", action="store_true", help="Process and save without cv2.imshow")
    parser.add_argument("--snapshot-every", type=int, default=60, help="Save one monitoring screenshot every N frames")
    args = parser.parse_args()
    if not 0 < args.gamma < 1 or args.snapshot_every < 1:
        raise ValueError("Gamma must be 0-1 and --snapshot-every must be positive.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = args.video or first_video(DATA)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUTPUT / "side_by_side_enhanced.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    frame_number = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        processed = enhance(raw, args.gamma)
        monitor = cv2.hconcat([raw, processed])
        cv2.putText(monitor, "Raw", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(monitor, "Enhanced", (width + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(monitor)
        if frame_number % args.snapshot_every == 0:
            cv2.imwrite(str(OUTPUT / f"monitor_{frame_number:05d}.png"), monitor)
        if not args.no_display:
            cv2.imshow("Task 3 - raw (left) / enhanced (right); q to quit", monitor)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break
        frame_number += 1
    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"Processed {frame_number} frames. Saved results in {OUTPUT}")


if __name__ == "__main__":
    main()
