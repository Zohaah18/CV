import os
import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import joblib
from torchvision import models
from torchvision.datasets import VOCDetection
from torchvision.ops import nms, box_iou
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CLASSES = ["cat", "dog", "car", "bus", "bicycle"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="rcnn_output")
    p.add_argument("--train_images", type=int, default=150)
    p.add_argument("--test_images", type=int, default=30)
    p.add_argument("--train_proposals", type=int, default=300)
    p.add_argument("--test_proposals", type=int, default=500)
    p.add_argument("--pos_per_image", type=int, default=16)
    p.add_argument("--neg_per_image", type=int, default=48)
    p.add_argument("--score_thresh", type=float, default=0.0)
    p.add_argument("--nms_thresh", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_samples(dataset, count, used):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    samples = []
    for idx in indices:
        if idx in used:
            continue
        image, target = dataset[idx]
        objects = target["annotation"]["object"]
        if isinstance(objects, dict):
            objects = [objects]
        boxes, labels = [], []
        for obj in objects:
            name = obj["name"]
            if name not in CLASSES:
                continue
            bb = obj["bndbox"]
            boxes.append([float(bb["xmin"]), float(bb["ymin"]), float(bb["xmax"]), float(bb["ymax"])])
            labels.append(CLASSES.index(name) + 1)
        if not boxes:
            continue
        used.add(idx)
        samples.append({
            "image": np.array(image.convert("RGB")),
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "id": target["annotation"]["filename"],
        })
        if len(samples) >= count:
            break
    return samples


def selective_search(image_rgb, max_boxes):
    cv2.setUseOptimized(True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(bgr)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    boxes = []
    for x, y, w, h in rects:
        if w < 20 or h < 20:
            continue
        boxes.append([x, y, x + w, y + h])
        if len(boxes) >= max_boxes:
            break
    return np.array(boxes, dtype=np.float32)


def build_feature_extractor(device):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = nn.Identity()
    net.eval()
    return net.to(device)


@torch.no_grad()
def extract_features(image_rgb, boxes, net, device, batch_size=64):
    h, w = image_rgb.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, x2)), int(min(h, y2))
        crop = image_rgb[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crop = (crop.astype(np.float32) / 255.0 - MEAN) / STD
        crops.append(crop.transpose(2, 0, 1))
    feats = []
    for i in range(0, len(crops), batch_size):
        batch = torch.from_numpy(np.stack(crops[i:i + batch_size])).to(device)
        feats.append(net(batch).cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)


def encode_deltas(props, gts):
    pw = props[:, 2] - props[:, 0]
    ph = props[:, 3] - props[:, 1]
    px = props[:, 0] + 0.5 * pw
    py = props[:, 1] + 0.5 * ph
    gw = gts[:, 2] - gts[:, 0]
    gh = gts[:, 3] - gts[:, 1]
    gx = gts[:, 0] + 0.5 * gw
    gy = gts[:, 1] + 0.5 * gh
    return np.stack([(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)], axis=1)


def decode_deltas(props, deltas, width, height):
    pw = props[:, 2] - props[:, 0]
    ph = props[:, 3] - props[:, 1]
    px = props[:, 0] + 0.5 * pw
    py = props[:, 1] + 0.5 * ph
    dw = np.clip(deltas[:, 2], -1.0, 1.0)
    dh = np.clip(deltas[:, 3], -1.0, 1.0)
    cx = px + deltas[:, 0] * pw
    cy = py + deltas[:, 1] * ph
    bw = pw * np.exp(dw)
    bh = ph * np.exp(dh)
    out = np.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], axis=1)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height - 1)
    return out


def iou_matrix(a, b):
    return box_iou(torch.from_numpy(a), torch.from_numpy(b)).numpy()


def build_training_set(samples, net, device, args):
    all_feats, all_labels = [], []
    reg_feats, reg_targets = [], []
    for n, s in enumerate(samples):
        props = selective_search(s["image"], args.train_proposals)
        props = np.concatenate([props, s["boxes"]], axis=0) if len(props) else s["boxes"].copy()
        ious = iou_matrix(props, s["boxes"])
        best = ious.max(axis=1)
        arg = ious.argmax(axis=1)
        pos_idx = np.where(best >= 0.5)[0]
        neg_idx = np.where(best < 0.3)[0]
        if len(pos_idx) > args.pos_per_image:
            pos_idx = np.random.choice(pos_idx, args.pos_per_image, replace=False)
        if len(neg_idx) > args.neg_per_image:
            neg_idx = np.random.choice(neg_idx, args.neg_per_image, replace=False)
        chosen = np.concatenate([pos_idx, neg_idx]).astype(int)
        if len(chosen) == 0:
            continue
        feats = extract_features(s["image"], props[chosen], net, device)
        labels = np.zeros(len(chosen), dtype=np.int64)
        labels[:len(pos_idx)] = s["labels"][arg[pos_idx]]
        all_feats.append(feats)
        all_labels.append(labels)
        if len(pos_idx):
            reg_feats.append(feats[:len(pos_idx)])
            reg_targets.append(encode_deltas(props[pos_idx], s["boxes"][arg[pos_idx]]))
        print(f"train image {n + 1}/{len(samples)}: {len(pos_idx)} positive, {len(neg_idx)} negative")
    return (
        np.concatenate(all_feats),
        np.concatenate(all_labels),
        np.concatenate(reg_feats),
        np.concatenate(reg_targets),
    )


def detect(image, net, device, scaler, svm, regressor, args):
    h, w = image.shape[:2]
    props = selective_search(image, args.test_proposals)
    if len(props) == 0:
        return []
    feats = scaler.transform(extract_features(image, props, net, device))
    scores = svm.decision_function(feats)
    deltas = regressor.predict(feats)
    refined = decode_deltas(props, deltas, w, h)
    classes = list(svm.classes_)
    detections = []
    for k in range(1, len(CLASSES) + 1):
        if k not in classes:
            continue
        s = scores[:, classes.index(k)]
        keep = np.where(s > args.score_thresh)[0]
        if len(keep) == 0:
            continue
        b = torch.from_numpy(refined[keep]).float()
        sc = torch.from_numpy(s[keep]).float()
        kept = nms(b, sc, args.nms_thresh).numpy()
        for j in kept:
            detections.append((k, float(sc[j]), refined[keep][j]))
    return detections


def average_precision(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(all_dets, samples):
    aps = {}
    for k, name in enumerate(CLASSES, start=1):
        gt_per_image = {}
        total = 0
        for i, s in enumerate(samples):
            mask = s["labels"] == k
            gt_per_image[i] = {"boxes": s["boxes"][mask], "used": np.zeros(mask.sum(), dtype=bool)}
            total += int(mask.sum())
        dets = []
        for i, dlist in enumerate(all_dets):
            for label, score, box in dlist:
                if label == k:
                    dets.append((i, score, box))
        if total == 0:
            continue
        dets.sort(key=lambda d: -d[1])
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for n, (i, score, box) in enumerate(dets):
            g = gt_per_image[i]
            if len(g["boxes"]) == 0:
                fp[n] = 1
                continue
            ious = iou_matrix(box[None].astype(np.float32), g["boxes"])[0]
            j = int(ious.argmax())
            if ious[j] >= 0.5 and not g["used"][j]:
                tp[n] = 1
                g["used"][j] = True
            else:
                fp[n] = 1
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec = tp_c / total
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        aps[name] = average_precision(rec, prec) if len(dets) else 0.0
    return aps


def draw_detections(image, detections, path):
    canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    palette = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255)]
    for label, score, box in detections:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = palette[(label - 1) % len(palette)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{CLASSES[label - 1]} {score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    if not hasattr(cv2, "ximgproc"):
        raise SystemExit("selective search needs opencv-contrib-python: pip install opencv-contrib-python")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = VOCDetection(args.data_root, year="2007", image_set="trainval", download=True)
    used = set()
    train_samples = load_samples(dataset, args.train_images, used)
    test_samples = load_samples(dataset, args.test_images, used)
    print(f"train images: {len(train_samples)}, test images: {len(test_samples)}")

    net = build_feature_extractor(device)

    feats, labels, reg_feats, reg_targets = build_training_set(train_samples, net, device, args)
    print("training samples:", feats.shape, "regression samples:", reg_feats.shape)

    scaler = StandardScaler().fit(feats)
    svm = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)
    svm.fit(scaler.transform(feats), labels)
    regressor = Ridge(alpha=1.0)
    regressor.fit(scaler.transform(reg_feats), reg_targets)

    joblib.dump({"scaler": scaler, "svm": svm, "regressor": regressor},
                os.path.join(args.out_dir, "rcnn_heads.joblib"))

    all_dets = []
    for n, s in enumerate(test_samples):
        dets = detect(s["image"], net, device, scaler, svm, regressor, args)
        all_dets.append(dets)
        draw_detections(s["image"], dets, os.path.join(pred_dir, f"pred_{n:03d}_{s['id']}"))
        print(f"test image {n + 1}/{len(test_samples)}: {len(dets)} detections")

    aps = evaluate(all_dets, test_samples)
    for name, ap in aps.items():
        print(f"AP@0.5 {name}: {ap:.3f}")
    if aps:
        print(f"mAP@0.5: {np.mean(list(aps.values())):.3f}")
    print("predictions saved to", pred_dir)


if __name__ == "__main__":
    main()
