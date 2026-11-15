import argparse
import csv
import json
import platform
import random
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

OUT_DIR = Path("outputs_q3")
IMG_DIR = Path("sample_images")
COCO_VAL_IDS = [39769, 139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490]
COCO_URLS = [f"http://images.cocodataset.org/val2017/{i:012d}.jpg" for i in COCO_VAL_IDS]
FALLBACK_URLS = [
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
    "https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg",
]

GROUPS = {
    "person": {"person"},
    "vehicle": {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"},
    "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
    "street objects": {"traffic light", "fire hydrant", "stop sign", "parking meter", "bench"},
    "indoor/household": {"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                         "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                         "toothbrush"},
    "food & tableware": {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"},
    "accessories & sports": {"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                             "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                             "skateboard", "surfboard", "tennis racket"},
}
CLASS_TO_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}
SMALL_AREA = 32 * 32      

def _download(url, dest, timeout=20):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=timeout).read()
        if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"   (download failed: {url.split('/')[-1]} -> {type(e).__name__})")
        return False


def get_random_images(n, seed=None):
    rng = random.Random(seed)
    IMG_DIR.mkdir(exist_ok=True)
    pool = COCO_URLS[:]
    rng.shuffle(pool)
    paths = []
    for url in pool + FALLBACK_URLS:
        if len(paths) >= n:
            break
        dest = IMG_DIR / url.split("/")[-1]
        if dest.exists() or _download(url, dest):
            paths.append(dest)
    while len(paths) < n:
        url = f"https://picsum.photos/seed/{rng.randint(0, 10**6)}/640/480"
        dest = IMG_DIR / f"picsum_{url.split('/')[-3]}.jpg"
        if dest.exists() or _download(url, dest):
            paths.append(dest)
        else:
            break
    cached = sorted(IMG_DIR.glob("*.jp*g"))
    if len(paths) < n and cached:
        paths += rng.sample(cached, min(n - len(paths), len(cached)))
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise SystemExit("No images available. Put .jpg files in ./sample_images or use --source webcam")
    return imgs[:n]


def grab_webcam_frames(n, camera=0):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")
    frames = []
    print(f"Capturing {n} webcam frames (move objects into view) ...")
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        time.sleep(0.25)
    cap.release()
    return frames


class YOLOv3Detector:
    name = "YOLOv3"

    def __init__(self, conf, device, tiny=False):
        from ultralytics import YOLO
        self.weights = "yolov3-tinyu.pt" if tiny else "yolov3u.pt"
        if tiny:
            self.name = "YOLOv3-tiny"
        self.model = YOLO(self.weights)
        self.conf, self.device = conf, device
        self.params_m = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        ck = getattr(self.model, "ckpt_path", None)
        self.file_mb = Path(ck).stat().st_size / 1e6 if ck and Path(ck).exists() else None
        self.use_cuda = str(device).startswith("cuda")

    def detect(self, bgr):
        r = self.model(bgr, conf=self.conf, device=self.device, verbose=False)[0]
        return [(*b.xyxy[0].tolist(), r.names[int(b.cls[0])], float(b.conf[0])) for b in r.boxes]


class TorchvisionDetector:
    def __init__(self, name, conf, device):
        import torch
        import torchvision.models.detection as det
        self.torch, self.name, self.device = torch, name, device
        if name == "SSD":
            w = det.SSD300_VGG16_Weights.DEFAULT
            self.model = det.ssd300_vgg16(weights=w, score_thresh=conf)
        else:
            w = det.RetinaNet_ResNet50_FPN_Weights.DEFAULT
            self.model = det.retinanet_resnet50_fpn(weights=w, score_thresh=conf)
        self.model.to(device).eval()
        self.categories = w.meta["categories"]
        self.params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        f = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(w.url).path).name
        self.file_mb = f.stat().st_size / 1e6 if f.exists() else None
        self.use_cuda = device.type == "cuda"

    def detect(self, bgr):
        torch = self.torch
        t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model([t])[0]
        return [(*b, self.categories[l], float(s)) for b, l, s in
                zip(out["boxes"].cpu().tolist(), out["labels"].cpu().tolist(), out["scores"].cpu().tolist())]


def build(label, conf, device, tiny):
    import torch
    dev_tv = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    dev_yolo = "cuda:0" if dev_tv.type == "cuda" else "cpu"
    if label == "YOLOv3":
        return YOLOv3Detector(conf, dev_yolo, tiny)
    return TorchvisionDetector(label, conf, dev_tv)


def sync(model):
    if model.use_cuda:
        import torch
        torch.cuda.synchronize()

def benchmark(label, frames, conf, device, tiny, repeats, warmup):
    t0 = time.perf_counter()
    m = build(label, conf, device, tiny)
    load_s = time.perf_counter() - t0
    print(f"\n[{m.name}] loaded in {load_s:.1f}s  | {m.params_m:.1f}M params")

    for i in range(warmup):                              
        m.detect(frames[i % len(frames)])

    times, dets_per_frame = [], []
    cls_conf, grp_conf = defaultdict(list), defaultdict(list)
    small_conf, small_n, total_n = [], 0, 0
    for r in range(repeats):
        for f in frames:
            sync(m)
            t = time.perf_counter()
            dets = m.detect(f)
            sync(m)
            times.append((time.perf_counter() - t) * 1000)
            if r == 0:                                  
                dets_per_frame.append(len(dets))
                for x1, y1, x2, y2, lab, sc in dets:
                    cls_conf[lab].append(sc)
                    grp_conf[CLASS_TO_GROUP.get(lab, "other")].append(sc)
                    total_n += 1
                    if (x2 - x1) * (y2 - y1) < SMALL_AREA:
                        small_n += 1
                        small_conf.append(sc)
    mean_ms = statistics.mean(times)
    all_conf = [c for v in cls_conf.values() for c in v]
    res = {
        "model": m.name,
        "device": str(m.device if hasattr(m, "device") else device),
        "load_time_s": round(load_s, 2),
        "params_millions": round(m.params_m, 2),
        "size_fp32_mb": round(m.params_m * 4, 1),
        "weights_file_mb": round(m.file_mb, 1) if m.file_mb else None,
        "n_frames": len(frames), "n_timed_runs": len(times),
        "latency_ms_mean": round(mean_ms, 1),
        "latency_ms_median": round(statistics.median(times), 1),
        "latency_ms_std": round(statistics.pstdev(times), 1),
        "latency_ms_p95": round(float(np.percentile(times, 95)), 1),
        "latency_ms_min": round(min(times), 1),
        "fps": round(1000 / mean_ms, 2),
        "total_detections": total_n,
        "avg_detections_per_frame": round(statistics.mean(dets_per_frame), 2),
        "mean_confidence": round(statistics.mean(all_conf), 3) if all_conf else None,
        "small_object_detections": small_n,
        "small_object_mean_conf": round(statistics.mean(small_conf), 3) if small_conf else None,
        "confidence_by_group": {g: {"mean": round(statistics.mean(v), 3), "n": len(v)} for g, v in grp_conf.items()},
        "confidence_by_class": {c: {"mean": round(statistics.mean(v), 3), "n": len(v)} for c, v in cls_conf.items()},
    }
    print(f"[{m.name}] {mean_ms:.1f} ms/frame  ->  {res['fps']:.2f} FPS | "
          f"{total_n} detections, mean conf {res['mean_confidence']}")
    del m
    return res

def print_table(results):
    cols = [("Model", "model", 12), ("FPS", "fps", 7), ("ms/frame", "latency_ms_mean", 9),
            ("p95 ms", "latency_ms_p95", 8), ("Params(M)", "params_millions", 10),
            ("Size MB", "size_fp32_mb", 8), ("Detections", "total_detections", 11),
            ("Mean conf", "mean_confidence", 10), ("Small objs", "small_object_detections", 11)]
    print("\n" + "=" * 100)
    print("".join(f"{h:<{w}}" for h, _, w in cols))
    print("-" * 100)
    for r in results:
        print("".join(f"{str(r[k]):<{w}}" for _, k, w in cols))
    print("=" * 100)
    groups = sorted({g for r in results for g in r["confidence_by_group"]})
    print("\nMean confidence by object type (n detections in brackets)")
    print(f"{'type':<24}" + "".join(f"{r['model']:<20}" for r in results))
    for g in groups:
        row = f"{g:<24}"
        for r in results:
            v = r["confidence_by_group"].get(g)
            row += f"{(str(v['mean']) + ' (' + str(v['n']) + ')') if v else '-':<20}"
        print(row)


def save_outputs(results, meta):
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    keys = [k for k in results[0] if not k.startswith("confidence_by")]
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})

    classes = sorted({c for r in results for c in r["confidence_by_class"]})
    with open(OUT_DIR / "per_class_confidence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + [f"{r['model']}_mean_conf" for r in results] + [f"{r['model']}_count" for r in results])
        for c in classes:
            w.writerow([c] + [r["confidence_by_class"].get(c, {}).get("mean", "") for r in results]
                       + [r["confidence_by_class"].get(c, {}).get("n", 0) for r in results])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping chart (pip install matplotlib)")
        return
    names = [r["model"] for r in results]
    colors = ["#2e9e44", "#e8890c", "#3b5bdb"][:len(names)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("Speed (FPS, higher is better)", "fps"), ("Inference time per frame (ms)", "latency_ms_mean"),
              ("Model size (MB, FP32)", "size_fp32_mb"), ("Mean confidence", "mean_confidence"),
              ("Small-object detections", "small_object_detections")]
    for a, (title, key) in zip(ax.flat, panels):
        vals = [r[key] or 0 for r in results]
        bars = a.bar(names, vals, color=colors)
        a.set_title(title)
        a.bar_label(bars, fmt="%.2f" if key in ("fps", "mean_confidence") else "%.0f")
    groups = sorted({g for r in results for g in r["confidence_by_group"]})
    a = ax.flat[5]
    x = np.arange(len(groups))
    wd = 0.8 / len(results)
    for i, r in enumerate(results):
        a.bar(x + i * wd, [r["confidence_by_group"].get(g, {}).get("mean", 0) for g in groups], wd,
              label=r["model"], color=colors[i])
    a.set_xticks(x + wd * (len(results) - 1) / 2)
    a.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    a.set_title("Mean confidence by object type")
    a.legend()
    fig.suptitle(f"Q3 benchmark  |  {meta['cpu']}  |  device: {meta['device']}  |  {meta['n_frames']} frames")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "benchmark.png", dpi=130)
    print(f"Chart saved: {OUT_DIR / 'benchmark.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["images", "webcam"], default="images")
    ap.add_argument("--n", type=int, default=6, help="number of random pictures")
    ap.add_argument("--frames", type=int, default=15, help="webcam frames to capture")
    ap.add_argument("--repeats", type=int, default=3, help="timed passes over all frames")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--tiny", action="store_true", help="YOLOv3-tiny instead of YOLOv3")
    ap.add_argument("--models", nargs="+", default=["YOLOv3", "SSD", "RetinaNet"],
                    choices=["YOLOv3", "SSD", "RetinaNet"])
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    if args.source == "webcam":
        frames = grab_webcam_frames(args.frames, args.camera)
    else:
        print("Fetching random pictures ...")
        frames = get_random_images(args.n, args.seed)
    print(f"Benchmarking on {len(frames)} frames, {args.repeats} timed passes each.")

    results = [benchmark(m, frames, args.conf, args.device, args.tiny, args.repeats, args.warmup)
               for m in args.models]

    print_table(results)
    meta = {"cpu": platform.processor() or platform.machine(), "platform": platform.platform(),
            "device": args.device, "n_frames": len(frames), "conf_threshold": args.conf,
            "source": args.source}
    save_outputs(results, meta)
    print(f"\nSaved results to ./{OUT_DIR}/  (run q4_discussion.py next)")


if __name__ == "__main__":
    main()