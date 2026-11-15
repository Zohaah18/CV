import argparse
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs_q2")
IMG_DIR = Path("sample_images")

COCO_VAL_IDS = [39769, 139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490]
COCO_URLS = [f"http://images.cocodataset.org/val2017/{i:012d}.jpg" for i in COCO_VAL_IDS]
FALLBACK_URLS = [
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
    "https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg",
]


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
    paths = []
    pool = COCO_URLS[:]
    rng.shuffle(pool)
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
    imgs = [(p.name, cv2.imread(str(p))) for p in paths]
    imgs = [(nm, im) for nm, im in imgs if im is not None]
    if not imgs:
        raise SystemExit("No images available (offline?). Put some .jpg files in ./sample_images "
                         "or pass --images a.jpg b.jpg")
    return imgs[:n]


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

    def detect(self, bgr):
        torch = self.torch
        t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model([t])[0]
        res = []
        for (x1, y1, x2, y2), lab, sc in zip(out["boxes"].cpu().tolist(),
                                              out["labels"].cpu().tolist(),
                                              out["scores"].cpu().tolist()):
            res.append((x1, y1, x2, y2, self.categories[lab], float(sc)))
        return res


def load_models(conf, device, tiny):
    import torch
    dev_tv = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    dev_yolo = "cuda:0" if dev_tv.type == "cuda" else "cpu"
    models = []
    for label, factory in [("YOLOv3", lambda: YOLOv3Detector(conf, dev_yolo, tiny)),
                           ("SSD", lambda: TorchvisionDetector("SSD", conf, dev_tv)),
                           ("RetinaNet", lambda: TorchvisionDetector("RetinaNet", conf, dev_tv))]:
        print(f"Loading {label} ...")
        try:
            models.append(factory())
        except Exception as e:   
            print(f"   !! could not load {label}: {type(e).__name__}: {str(e)[:150]}")
    if not models:
        raise SystemExit("No model could be loaded - check your internet connection (weights download on first run).")
    return models

PALETTE = {"YOLOv3": (0, 200, 0), "YOLOv3-tiny": (0, 200, 0), "SSD": (255, 140, 0), "RetinaNet": (60, 60, 255)}


def draw(img, dets, color, header=None):
    out = img.copy()
    for x1, y1, x2, y2, label, score in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(out, txt, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if header:
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_h(img, h=480):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def show(title, img, wait=0):
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return -1   
    try:
        cv2.imshow(title, img)
        return cv2.waitKey(wait) & 0xFF
    except cv2.error:
        return -1   


def run_images(models, images, no_show):
    OUT_DIR.mkdir(exist_ok=True)
    for name, img in images:
        stem = Path(name).stem
        panels = []
        print(f"\n=== {name}  ({img.shape[1]}x{img.shape[0]}) ===")
        for m in models:
            m.detect(img)                                    
            t0 = time.perf_counter()
            dets = m.detect(img)
            ms = (time.perf_counter() - t0) * 1000
            print(f"{m.name:12s} {len(dets):3d} objects   {ms:8.1f} ms")
            for x1, y1, x2, y2, lab, sc in sorted(dets, key=lambda d: -d[5])[:8]:
                print(f"     {lab:15s} conf={sc:.2f}  box=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
            ann = draw(img, dets, PALETTE[m.name], f"{m.name}: {len(dets)} objects, {ms:.0f} ms")
            cv2.imwrite(str(OUT_DIR / f"{stem}_{m.name}.jpg"), ann)
            panels.append(resize_h(ann))
        montage = np.hstack(panels)
        cv2.imwrite(str(OUT_DIR / f"{stem}_comparison.jpg"), montage)
        if not no_show:
            show(f"Q2 comparison: {name}  (any key = next)", resize_h(montage, 420))
    cv2.destroyAllWindows()
    print(f"\nSaved annotated images to ./{OUT_DIR}/")


def run_stream(models, cap):
    idx, fps, prev = 0, 0.0, time.time()
    print("Keys: 1=YOLOv3  2=SSD  3=RetinaNet  q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = models[idx]
        dets = m.detect(frame)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        out = draw(frame, dets, PALETTE[m.name], f"{m.name} | {fps:.1f} FPS | {len(dets)} objects")
        key = show("Q2 - YOLOv3 / SSD / RetinaNet", out, wait=1)
        if key == ord("q"):
            break
        if key in (ord("1"), ord("2"), ord("3")):
            idx = min(key - ord("1"), len(models) - 1)
            fps = 0.0
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["images", "webcam", "video"], default="images")
    ap.add_argument("--images", nargs="*", help="your own image paths (default: random pictures)")
    ap.add_argument("--n", type=int, default=4, help="number of random pictures")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--tiny", action="store_true", help="use YOLOv3-tiny (much faster on CPU)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    models = load_models(args.conf, args.device, args.tiny)

    if args.source == "images":
        if args.images:
            images = [(Path(p).name, cv2.imread(p)) for p in args.images]
            images = [(n, i) for n, i in images if i is not None]
        else:
            print("Fetching random pictures ...")
            images = get_random_images(args.n, args.seed)
        run_images(models, images, args.no_show)
    else:
        cap = cv2.VideoCapture(args.video if args.source == "video" else args.camera)
        if not cap.isOpened():
            raise SystemExit("Could not open the video source.")
        run_stream(models, cap)


if __name__ == "__main__":
    main()