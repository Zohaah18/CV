"""Task 2: histogram-enhanced, weighted CT/MRI image fusion."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA, OUTPUT = ROOT / "data", ROOT / "output"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def candidate(folder: Path, word: str) -> Path:
    found = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS and word in p.stem.lower())
    if not found:
        raise FileNotFoundError(f"No file containing '{word}' in {folder}; pass --{word} explicitly.")
    return found[0]


def log_transform(image: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(255 / np.log1p(255) * np.log1p(image.astype(np.float32)), 0, 255))


def gamma_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    return np.uint8(np.clip(255 * (image.astype(np.float32) / 255) ** gamma, 0, 255))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, help="Matched CT slice")
    parser.add_argument("--mri", type=Path, help="Matched MRI slice")
    parser.add_argument("--ct-weight", type=float, default=0.65, help="CT contribution; must exceed MRI")
    parser.add_argument("--gamma", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 < args.ct_weight < 1 or not 0 < args.gamma < 1:
        raise ValueError("Weights and gamma must be between 0 and 1.")
    mri_weight = 1 - args.ct_weight
    if args.ct_weight <= mri_weight:
        raise ValueError("The brief requires a heavier CT weight than MRI.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ct_path, mri_path = args.ct or candidate(DATA, "ct"), args.mri or candidate(DATA, "mri")
    ct = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
    mri = cv2.imread(str(mri_path), cv2.IMREAD_GRAYSCALE)
    if ct is None or mri is None:
        raise ValueError("OpenCV could not load one of the selected slices.")
    if ct.shape != mri.shape:
        print("Warning: resizing MRI to CT dimensions. Confirm the pair is already anatomically aligned.")
        mri = cv2.resize(mri, (ct.shape[1], ct.shape[0]), interpolation=cv2.INTER_LINEAR)

    ct_eq, mri_eq = cv2.equalizeHist(ct), cv2.equalizeHist(mri)
    ct_color = cv2.applyColorMap(ct_eq, cv2.COLORMAP_JET)
    mri_color = cv2.applyColorMap(mri_eq, cv2.COLORMAP_HOT)
    fused = cv2.addWeighted(ct_color, args.ct_weight, mri_color, mri_weight, 0)
    fused_gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    log_fused = log_transform(fused_gray)
    gamma_fused = gamma_transform(log_fused, args.gamma)
    final = cv2.applyColorMap(gamma_fused, cv2.COLORMAP_JET)

    for name, image in {"01_ct_equalized.png": ct_eq, "02_mri_equalized.png": mri_eq,
                        "03_ct_heatmap.png": ct_color, "04_mri_heatmap.png": mri_color,
                        "05_weighted_fusion.png": fused, "06_log_fusion.png": log_fused,
                        "07_final_gamma_fusion.png": final}.items():
        cv2.imwrite(str(OUTPUT / name), image)
    (OUTPUT / "metrics.txt").write_text(
        f"CT weight: {args.ct_weight:.2f}\nMRI weight: {mri_weight:.2f}\n"
        f"CT equalized mean/std: {ct_eq.mean():.2f}/{ct_eq.std():.2f}\n"
        f"MRI equalized mean/std: {mri_eq.mean():.2f}/{mri_eq.std():.2f}\n"
        f"Fused grayscale mean/std: {fused_gray.mean():.2f}/{fused_gray.std():.2f}\n", encoding="utf-8")

    panels = [("CT (equalized)", ct_eq, "gray"), ("MRI (equalized)", mri_eq, "gray"),
              ("CT heatmap", cv2.cvtColor(ct_color, cv2.COLOR_BGR2RGB), None),
              ("MRI heatmap", cv2.cvtColor(mri_color, cv2.COLOR_BGR2RGB), None),
              (f"Weighted fusion: CT {args.ct_weight:.2f}, MRI {mri_weight:.2f}", cv2.cvtColor(fused, cv2.COLOR_BGR2RGB), None),
              ("Log + power-law fusion", cv2.cvtColor(final, cv2.COLOR_BGR2RGB), None)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.suptitle("Task 2: comparative CT / MRI fusion", fontsize=16)
    fig.tight_layout(); fig.savefig(OUTPUT / "comparison.png", dpi=180, bbox_inches="tight"); plt.show()
    print(f"Saved Task 2 results in {OUTPUT}")


if __name__ == "__main__":
    main()
