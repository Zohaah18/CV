import argparse
import json
from pathlib import Path

OUT_DIR = Path("outputs_q4")

LITERATURE = [
    ("YOLOv3-608",          33.0, 18.3, "~51 ms on Titan X (paper)"),
    ("SSD513",              31.2, 10.2, "~125 ms on Titan X (paper)"),
    ("RetinaNet-101-800",   37.8, 24.1, "~198 ms on Titan X (paper)"),
    ("torchvision SSD300 VGG16",           25.1, None, "35.6M params (torchvision docs)"),
    ("torchvision RetinaNet R50-FPN",      36.4, None, "34.0M params (torchvision docs)"),
]

ARCH = {
    "YOLOv3": (
        "Single-stage, Darknet-53 backbone with predictions at THREE scales (stride 32/16/8), so it keeps "
        "some resolution for small objects. One forward pass, no region proposals -> very fast. Weaker than "
        "RetinaNet on tight/small boxes, and the full model is large (~100M params in the Ultralytics port)."),
    "YOLOv3-tiny": (
        "Cut-down YOLOv3 (two scales, far fewer layers). Tiny footprint and the fastest of the three, "
        "but noticeably less accurate, especially on small or crowded objects."),
    "SSD": (
        "Single-stage, VGG-16 backbone, boxes predicted from several feature maps. Simple and quick, but the "
        "shallow maps that see small objects carry weak semantics, and the 300x300 input shrinks small "
        "objects further -> the usual weak spot is SMALL objects. VGG-16 makes it heavier than it looks."),
    "RetinaNet": (
        "Single-stage, ResNet-50 + Feature Pyramid Network, trained with FOCAL LOSS to handle the "
        "foreground/background imbalance. FPN gives semantically rich high-resolution maps -> best accuracy "
        "on small objects of the three. The price is a heavier head (dense anchors at 5 pyramid levels) "
        "and the lowest FPS."),
}

EXPECTED = {
    "real-time": ("YOLOv3-tiny", "YOLOv3"),        
    "small objects": ("RetinaNet",),
    "resource-limited": ("YOLOv3-tiny", "SSD"),
}


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def rank(results, key, reverse):
    rows = [r for r in results if r.get(key) is not None]
    return sorted(rows, key=lambda r: r[key], reverse=reverse)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_report(data):
    L = ["# Q4 - Which model performs best?", ""]
    results = data["results"] if data else None

    if results:
        meta = data["meta"]
        L += [f"Measured by Q3 on **{meta['n_frames']} frames**, device **{meta['device']}**, "
              f"confidence threshold {meta['conf_threshold']}.", ""]
        L += ["## 1. Measured results", ""]
        L.append(md_table(
            ["Model", "FPS", "ms/frame", "Size (MB, FP32)", "Detections", "Mean conf", "Small objs", "Small-obj conf"],
            [[r["model"], r["fps"], r["latency_ms_mean"], r["size_fp32_mb"], r["total_detections"],
              r["mean_confidence"], r["small_object_detections"], r["small_object_mean_conf"]] for r in results]))
        L.append("")

        fastest = rank(results, "fps", True)
        smallest = rank(results, "size_fp32_mb", False)
        small_best = sorted(results, key=lambda r: (r["small_object_detections"], r["small_object_mean_conf"] or 0),
                            reverse=True)
        conf_best = rank(results, "mean_confidence", True)

        L += ["## 2. Scenario analysis", ""]

        w = fastest[0]
        L += ["### (a) Real-time detection", "",
              f"**Measured winner: {w['model']}** ({w['fps']} FPS, {w['latency_ms_mean']} ms/frame). Ranking: "
              + " > ".join(f"{r['model']} ({r['fps']} FPS)" for r in fastest) + ".", "",
              "Real-time needs roughly 25-30 FPS (< ~40 ms per frame). "
              + ("At least one model reaches that on this machine." if w["fps"] >= 25 else
                 "No model reaches 25 FPS on this machine/device - on a GPU the order stays the same but the "
                 "numbers jump; on CPU use the tiny variant or a smaller input size."), "",
              f"Why: single-stage detectors skip region proposals; the one with the lightest backbone/head wins. "
              f"{ARCH.get(w['model'], '')}", ""]

        w = small_best[0]
        L += ["### (b) Small-object recognition", "",
              f"**Measured winner: {w['model']}** ({w['small_object_detections']} detections under 32x32 px, "
              f"mean conf {w['small_object_mean_conf']}). Ranking by small-object count: "
              + " > ".join(f"{r['model']} ({r['small_object_detections']})" for r in small_best) + ".", "",
              "Caveat: with no ground-truth labels, a higher count is a proxy for recall, not proof of correctness - "
              "a model can also add false positives. Literature (table in section 3) is the better judge of accuracy.",
              "", f"Why: {ARCH.get(w['model'], '')}", ""]

        w_size = smallest[0]
        cpu_fast = fastest[0]
        L += ["### (c) Resource-limited systems (CPU-only)", "",
              f"Smallest model: **{w_size['model']}** ({w_size['size_fp32_mb']} MB FP32). "
              f"Fastest on this run: **{cpu_fast['model']}** ({cpu_fast['fps']} FPS). "
              "For a CPU-only / embedded box what matters is the combination of memory footprint and latency:", "",
              md_table(["Model", "Size (MB)", "ms/frame", "FPS"],
                       [[r["model"], r["size_fp32_mb"], r["latency_ms_mean"], r["fps"]] for r in fastest]), "",
              "Why: " + ARCH.get(cpu_fast["model"], ""), ""]

        L += ["### Measured vs. expected from architecture", ""]
        exp_rows = []
        for scen, measured in [("real-time", fastest[0]["model"]), ("small objects", small_best[0]["model"]),
                               ("resource-limited", cpu_fast["model"])]:
            exp = EXPECTED[scen]
            ok = measured in exp
            exp_rows.append([scen, " / ".join(exp), measured, "matches" if ok else "differs"])
        L.append(md_table(["Scenario", "Expected", "Measured", ""], exp_rows))
        L += ["", "Where the result differs, likely reasons are: few/odd test images, a low confidence threshold "
              "letting weaker models emit extra boxes, CPU thread scheduling noise, or YOLOv3-tiny vs full "
              "YOLOv3 being used. Re-run Q3 with more frames (`--n 12 --repeats 5`) to firm it up.", ""]
        L += [f"Highest mean confidence: **{conf_best[0]['model']}** ({conf_best[0]['mean_confidence']}). "
              "Note that confidence scores are not calibrated across architectures, so don't read them as accuracy.",
              ""]
    else:
        L += ["> `outputs_q3/results.json` not found - run `python q3_measure_and_record.py` first.",
              "> Below is the architecture-based discussion only (NOT your measurements).", ""]
        L += ["## 1. Expected behaviour from architecture", ""]
        for k, v in ARCH.items():
            L += [f"- **{k}**: {v}"]
        L += ["", "## 2. Expected ranking", "",
              md_table(["Scenario", "Best choice", "Reason"], [
                  ["Real-time", "YOLOv3 (tiny variant for CPU)", "Single pass, multi-scale heads, lightest per-frame cost"],
                  ["Small objects", "RetinaNet", "FPN + focal loss keep resolution and semantics for small boxes"],
                  ["CPU-only / low resources", "YOLOv3-tiny (or SSD with a light backbone)", "Smallest footprint, lowest latency"]]),
              ""]

    L += ["## 3. Published accuracy numbers (for context, approximate - verify before citing)", "",
          md_table(["Model", "COCO AP", "AP small", "Note"],
                   [[n, ap, aps if aps is not None else "-", note] for n, ap, aps, note in LITERATURE]), "",
          "Reading: RetinaNet has the best accuracy and clearly the best small-object AP; YOLOv3 trades some "
          "accuracy for much better speed; SSD is mid-way on accuracy but weakest on small objects.", ""]

    L += ["## 4. Conclusion", "",
          "| Scenario | Best model | One-line reason |", "|---|---|---|",
          "| Real-time detection | **YOLOv3** (YOLOv3-tiny on CPU) | single pass, multi-scale, best speed/accuracy balance |",
          "| Small object recognition | **RetinaNet** | FPN + focal loss, highest small-object AP |",
          "| Resource-limited / CPU-only | **YOLOv3-tiny** (SSD as the alternative) | smallest size and lowest latency |",
          "", "There is no single winner: pick by constraint - speed (YOLO), small-object accuracy (RetinaNet), "
          "footprint (tiny/lightweight variants).", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs_q3/results.json")
    args = ap.parse_args()
    report = build_report(load(args.results))
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "discussion.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved: {OUT_DIR / 'discussion.md'}")


if __name__ == "__main__":
    main()
