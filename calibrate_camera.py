import cv2
import numpy as np
import glob
import argparse

def calibrate(image_dir, grid_size, square_size, img_max_width=1024):
    objp = np.zeros((grid_size[1] * grid_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points = []

    image_paths = glob.glob(f"{image_dir}/*.png")
    image_paths = sorted(image_paths)

    if not image_paths:
        print("No images found for calibration.")
        return

    img_shape = None
    for fname in image_paths:
        img = cv2.imread(fname)
        img = resize_image(img, img_max_width)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray.shape[::-1]
        else:
            assert gray.shape[::-1] == img_shape, "Inconsistent image dimensions"

        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)
        else:
            print(f"Chessboard corners not found in image {fname}")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

    if ret:
        total_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(obj_points)
        print(f"Calibration successful.")
        print(f"RMS re-projection error: {mean_error}")
        print(f"Camera matrix (K):\n{K}")
        print(f"Distortion coefficients:\n{dist}")
        np.savez('calibration_data.npz', K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print("Calibration data saved to 'calibration_data.npz'.")
    else:
        print("Calibration failed.")

def resize_image(img, max_width):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('--dir', type=str, default='.', help='Directory with calibration images')
    parser.add_argument('--grid', type=str, default='7x4', help='Chessboard grid size as NxM (inner corners)')
    parser.add_argument('--square_size', type=float, default=0.05, help='Size of a square in meters')
    parser.add_argument('--width', type=int, default=1024, help='Maximum image width')

    args = parser.parse_args()

    grid_size = tuple(map(int, args.grid.split('x')))
    calibrate(args.dir, grid_size, args.square_size, img_max_width=args.width)
