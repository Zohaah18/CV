import argparse
import glob
import os
import cv2
import numpy as np

CALIB_FILE = "camera_calibration.npz"


def calibrate_from_images(image_glob, board_size, square_size):
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    obj_points, img_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_size = None
    for path in sorted(glob.glob(image_glob)):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
    if len(obj_points) < 5:
        raise SystemExit("Need at least 5 images with a detected chessboard")
    return run_calibration(obj_points, img_points, image_size, len(obj_points))


def calibrate_from_webcam(cam_index, board_size, square_size, target_frames):
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    obj_points, img_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    image_size = None
    print("Show the chessboard from different angles. SPACE = capture, Q = finish.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, (cols, rows), corners, found)
        cv2.putText(view, f"Captured {len(obj_points)}/{target_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Calibration", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            print(f"Captured {len(obj_points)}")
            if len(obj_points) >= target_frames:
                break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(obj_points) < 5:
        raise SystemExit("Need at least 5 captured views")
    return run_calibration(obj_points, img_points, image_size, len(obj_points))


def run_calibration(obj_points, img_points, image_size, count):
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    total_error = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        total_error += cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
    print("Views used:", count)
    print("RMS reprojection error:", round(rms, 4))
    print("Mean reprojection error:", round(total_error / len(obj_points), 4))
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:", dist.ravel())
    np.savez(CALIB_FILE, camera_matrix=mtx, dist_coeffs=dist, image_size=np.array(image_size))
    print("Saved to", CALIB_FILE)
    return mtx, dist


def load_calibration():
    data = np.load(CALIB_FILE)
    return data["camera_matrix"], data["dist_coeffs"]


def undistort_live(cam_index, alpha):
    mtx, dist = load_calibration()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Cannot read from webcam")
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2)
    use_remap = True
    crop = False
    print("Keys: Q quit, C toggle crop, M toggle cv2.undistort / remap")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if use_remap:
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        else:
            undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
        if crop:
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                undistorted = undistorted[y:y + rh, x:x + rw]
                undistorted = cv2.resize(undistorted, (w, h))
        label = "remap" if use_remap else "cv2.undistort"
        cv2.putText(undistorted, f"Undistorted ({label})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Live: original | undistorted", np.hstack([frame, undistorted]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            crop = not crop
        if key == ord("m"):
            use_remap = not use_remap
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "run", "all"], default="all")
    parser.add_argument("--images", default=None)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square", type=float, default=0.025)
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    board = (args.cols, args.rows)
    need_calib = args.mode in ("calibrate", "all") or not os.path.exists(CALIB_FILE)
    if need_calib:
        if args.images:
            calibrate_from_images(args.images, board, args.square)
        else:
            calibrate_from_webcam(args.camera, board, args.square, args.frames)
    if args.mode in ("run", "all"):
        undistort_live(args.camera, args.alpha)


if __name__ == "__main__":
    main()