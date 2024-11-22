import cv2
import numpy as np
import glob

# Checkerboard dimensions
checkerboard_size = (7, 4)  # Number of internal corners per chessboard row and column
square_size = 0.05  # Size of a square in meters

# Prepare object points based on the real-world dimensions of the checkerboard
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
obj_points = []  # 3D points in real-world space
img_points = []  # 2D points in image plane

# Load images using glob
image_files = glob.glob('calib_img_crs97/*.png')  # Adjust the path to your images

# Iterate over the image files
for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        # Refine corner positions to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners_refined)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners_refined, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard corners not found in image {fname}")

cv2.destroyAllWindows()

# Perform camera calibration if sufficient points have been collected
if len(obj_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    # Save the calibration results
    np.save("camera_matrix.npy", camera_matrix)
    np.save("distortion_coeffs.npy", dist_coeffs)

    # Output the calibration results
    print(f"RMS re-projection error: {ret}")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
else:
    print("Calibration failed: No chessboard corners were detected in any image.")
