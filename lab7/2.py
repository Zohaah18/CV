import argparse
import cv2
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def apply_perspective_transform(image, tilted_points, output_size=None):
    src = order_points(tilted_points)
    if output_size is None:
        w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
        h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    else:
        w, h = output_size
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h)), matrix


def auto_detect_document(image):
    scale = 800.0 / max(image.shape[:2])
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    ratio = 1.0 / (scale if scale < 1 else 1.0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32) * ratio
    return None


def pick_points_manually(image):
    points = []
    display = image.copy()
    scale = min(1.0, 900.0 / max(image.shape[:2]))
    display = cv2.resize(display, None, fx=scale, fy=scale)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Click 4 corners, then press Enter", display)

    cv2.imshow("Click 4 corners, then press Enter", display)
    cv2.setMouseCallback("Click 4 corners, then press Enter", on_mouse)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 4:
            break
        if key == 27:
            break
    cv2.destroyAllWindows()
    return np.float32(points) if len(points) == 4 else None


def make_demo_image():
    w, h = 600, 800
    page = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(page, (10, 10), (w - 10, h - 10), (150, 40, 40), 8)
    cv2.putText(page, "DOCUMENT", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)
    for i in range(12):
        y = 220 + i * 45
        cv2.line(page, (60, y), (w - 60, y), (70, 70, 70), 3)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[160, 100], [640, 40], [720, 580], [70, 520]])
    m = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((650, 800, 3), 120, np.uint8)
    out = cv2.warpPerspective(page, m, (800, 650), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--output", default="q2_unwarped.png")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    image = cv2.imread(args.image) if args.image else None
    if image is None:
        image = make_demo_image()

    points = pick_points_manually(image) if args.manual else auto_detect_document(image)
    if points is None:
        points = pick_points_manually(image)
    if points is None:
        raise SystemExit("No corner points available")

    size = (args.width, args.height) if args.width and args.height else None
    result, matrix = apply_perspective_transform(image, points, size)
    print("Inverse perspective matrix:\n", np.round(matrix, 5))

    preview = image.copy()
    cv2.polylines(preview, [np.int32(order_points(points))], True, (0, 0, 255), 3)
    cv2.imwrite("q2_detected.png", preview)
    cv2.imwrite(args.output, result)

    cv2.imshow("Tilted (source)", cv2.resize(preview, None, fx=min(1.0, 700.0 / preview.shape[1]), fy=min(1.0, 700.0 / preview.shape[1])))
    cv2.imshow("Unwarped (head-on)", cv2.resize(result, None, fx=min(1.0, 700.0 / result.shape[1]), fy=min(1.0, 700.0 / result.shape[1])))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()