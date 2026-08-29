from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def first_image(folder: Path) -> Path:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No image found in {folder}. Add a sample X-ray or pass --image.")
    return files[0]


def logarithmic(image: np.ndarray) -> np.ndarray:
    scale = 255.0 / np.log1p(255.0)
    return np.uint8(np.clip(scale * np.log1p(image.astype(np.float32)), 0, 255))


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((v / 255.0) ** gamma) * 255 for v in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def gray_world_balance(bgr: np.ndarray) -> np.ndarray:
    """Neutralize a colour cast by giving B, G, and R the same mean intensity."""
    values = bgr.astype(np.float32)
    means = values.mean(axis=(0, 1))
    target = float(means.mean())
    balanced = values * (target / np.maximum(means, 1e-6))
    return np.uint8(np.clip(balanced, 0, 255))


def save_gray(name: str, image: np.ndarray) -> None:
    cv2.imwrite(str(OUTPUT / name), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Relative or absolute path to a chest X-ray")
    parser.add_argument("--threshold", type=int, default=200, help="Dense-tissue threshold (0-255)")
    parser.add_argument("--gamma", type=float, default=0.6, help="Fractional gamma, must be below 1")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255 or not 0 < args.gamma < 1:
        raise ValueError("--threshold must be 0-255 and --gamma must be between 0 and 1.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = args.image or first_image(DATA)
    raw = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"OpenCV could not read {source}")

    equalized = cv2.equalizeHist(raw)
    heatmap = cv2.applyColorMap(equalized, cv2.COLORMAP_JET)
    balanced = gray_world_balance(heatmap)
    _, thresholded = cv2.threshold(equalized, args.threshold, 255, cv2.THRESH_BINARY)
    logged = logarithmic(raw)
    gamma = gamma_correct(raw, args.gamma)

    for name, image in {"01_raw.png": raw, "02_equalized.png": equalized,
                        "03_jet_heatmap.png": heatmap, "04_color_balanced.png": balanced,
                        "05_dense_tissue_mask.png": thresholded, "06_logarithmic.png": logged,
                        "07_gamma.png": gamma}.items():s
        save_gray(name, image)

    panels = [("Raw X-ray", raw, "gray"), ("Histogram equalization", equalized, "gray"),
              ("JET false color", cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), None),
              ("Color balance", cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB), None),
              (f"Dense tissue >= {args.threshold}", thresholded, "gray"),
              ("Logarithmic transform", logged, "gray"), (f"Gamma = {args.gamma}", gamma, "gray")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    axes.flat[-1].axis("off")
    fig.suptitle(f"Task 1: {source.name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved Task 1 results in {OUTPUT}")


if __name__ == "__main__":
    main()
