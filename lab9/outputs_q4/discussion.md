# Q4 - Which model performs best?

Measured by Q3 on **6 frames**, device **cpu**, confidence threshold 0.4.

## 1. Measured results

| Model | FPS | ms/frame | Size (MB, FP32) | Detections | Mean conf | Small objs | Small-obj conf |
|---|---|---|---|---|---|---|---|
| YOLOv3 | 0.51 | 1956.8 | 415.0 | 30 | 0.75 | 1 | 0.746 |
| SSD | 1.39 | 720.8 | 142.6 | 16 | 0.772 | 0 | None |
| RetinaNet | 0.24 | 4121.3 | 136.1 | 31 | 0.684 | 1 | 0.848 |

## 2. Scenario analysis

### (a) Real-time detection

**Measured winner: SSD** (1.39 FPS, 720.8 ms/frame). Ranking: SSD (1.39 FPS) > YOLOv3 (0.51 FPS) > RetinaNet (0.24 FPS).

Real-time needs roughly 25-30 FPS (< ~40 ms per frame). No model reaches 25 FPS on this machine/device - on a GPU the order stays the same but the numbers jump; on CPU use the tiny variant or a smaller input size.

Why: single-stage detectors skip region proposals; the one with the lightest backbone/head wins. Single-stage, VGG-16 backbone, boxes predicted from several feature maps. Simple and quick, but the shallow maps that see small objects carry weak semantics, and the 300x300 input shrinks small objects further -> the usual weak spot is SMALL objects. VGG-16 makes it heavier than it looks.

### (b) Small-object recognition

**Measured winner: RetinaNet** (1 detections under 32x32 px, mean conf 0.848). Ranking by small-object count: RetinaNet (1) > YOLOv3 (1) > SSD (0).

Caveat: with no ground-truth labels, a higher count is a proxy for recall, not proof of correctness - a model can also add false positives. Literature (table in section 3) is the better judge of accuracy.

Why: Single-stage, ResNet-50 + Feature Pyramid Network, trained with FOCAL LOSS to handle the foreground/background imbalance. FPN gives semantically rich high-resolution maps -> best accuracy on small objects of the three. The price is a heavier head (dense anchors at 5 pyramid levels) and the lowest FPS.

### (c) Resource-limited systems (CPU-only)

Smallest model: **RetinaNet** (136.1 MB FP32). Fastest on this run: **SSD** (1.39 FPS). For a CPU-only / embedded box what matters is the combination of memory footprint and latency:

| Model | Size (MB) | ms/frame | FPS |
|---|---|---|---|
| SSD | 142.6 | 720.8 | 1.39 |
| YOLOv3 | 415.0 | 1956.8 | 0.51 |
| RetinaNet | 136.1 | 4121.3 | 0.24 |

Why: Single-stage, VGG-16 backbone, boxes predicted from several feature maps. Simple and quick, but the shallow maps that see small objects carry weak semantics, and the 300x300 input shrinks small objects further -> the usual weak spot is SMALL objects. VGG-16 makes it heavier than it looks.

### Measured vs. expected from architecture

| Scenario | Expected | Measured |  |
|---|---|---|---|
| real-time | YOLOv3-tiny / YOLOv3 | SSD | differs |
| small objects | RetinaNet | RetinaNet | matches |
| resource-limited | YOLOv3-tiny / SSD | SSD | matches |

Where the result differs, likely reasons are: few/odd test images, a low confidence threshold letting weaker models emit extra boxes, CPU thread scheduling noise, or YOLOv3-tiny vs full YOLOv3 being used. Re-run Q3 with more frames (`--n 12 --repeats 5`) to firm it up.

Highest mean confidence: **SSD** (0.772). Note that confidence scores are not calibrated across architectures, so don't read them as accuracy.

## 3. Published accuracy numbers (for context, approximate - verify before citing)

| Model | COCO AP | AP small | Note |
|---|---|---|---|
| YOLOv3-608 | 33.0 | 18.3 | ~51 ms on Titan X (paper) |
| SSD513 | 31.2 | 10.2 | ~125 ms on Titan X (paper) |
| RetinaNet-101-800 | 37.8 | 24.1 | ~198 ms on Titan X (paper) |
| torchvision SSD300 VGG16 | 25.1 | - | 35.6M params (torchvision docs) |
| torchvision RetinaNet R50-FPN | 36.4 | - | 34.0M params (torchvision docs) |

Reading: RetinaNet has the best accuracy and clearly the best small-object AP; YOLOv3 trades some accuracy for much better speed; SSD is mid-way on accuracy but weakest on small objects.

## 4. Conclusion

| Scenario | Best model | One-line reason |
|---|---|---|
| Real-time detection | **YOLOv3** (YOLOv3-tiny on CPU) | single pass, multi-scale, best speed/accuracy balance |
| Small object recognition | **RetinaNet** | FPN + focal loss, highest small-object AP |
| Resource-limited / CPU-only | **YOLOv3-tiny** (SSD as the alternative) | smallest size and lowest latency |

There is no single winner: pick by constraint - speed (YOLO), small-object accuracy (RetinaNet), footprint (tiny/lightweight variants).
