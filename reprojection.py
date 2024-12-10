import cv2
import cv2.aruco as aruco
import numpy as np
import os
import re
from scipy.optimize import least_squares

# Load CSV file and parse it into a list of dictionaries
def load_csv(csv_path):
    with open(csv_path, "r") as file:
        lines = file.readlines()
        headers = lines[0].strip().split(",")
        data = [dict(zip(headers, line.strip().split(","))) for line in lines[1:]]
    return data

# Extract rotation and translation matrix from a CSV entry
def extract_robot_transformation(entry):
    T_bg = np.array([float(entry[f"T_bg_{i}"]) for i in range(16)]).reshape(4, 4)
    return T_bg[:3, :3], T_bg[:3, 3]  # Return rotation and translation

def match_csv_row(filename, csv_data):
    match = re.search(
        r'X([-+]?[0-9]*\.?[0-9]+)Y([-+]?[0-9]*\.?[0-9]+)Z([-+]?[0-9]*\.?[0-9]+)_J5([-+]?[0-9]*\.?[0-9]+)_J6([-+]?[0-9]*\.?[0-9]+)',
        filename
    )
    if match:
        # Extract values from the filename
        image_x = match.group(1)
        image_y = match.group(2)
        image_z = match.group(3)
        joint_5 = match.group(4)
        joint_6 = match.group(5)

        for row in csv_data:
            # Extract corresponding values from the CSV row
            csv_x = row["x"]
            csv_y = row["y"]
            csv_z = row["z"]
            csv_joint_5 = row["joint_5"]
            csv_joint_6 = row["joint_6"]

            # Check if the CSV values match directly with the image filename values
            if (
                image_x == csv_x and  # Tolerance for x
                image_y == csv_y and  # Tolerance for y
                image_z == csv_z and  # Tolerance for z
                csv_joint_5 == joint_5 and        # Check joint 5
                csv_joint_6 == joint_6            # Check joint 6
            ):
                return row
    raise ValueError(f"Cannot find matching row in CSV for filename: {filename}")

# Camera parameters (replace with actual values)
data = np.load('calibration_data.npz')
# camera_matrix = data['K']
camera_matrix = np.load("cameraMatrix.npy")
dist_coeffs = np.load("distCoeffs.npy")

# Paths to images and CSV
csv_path = "new_transformations.csv"  # Replace with the actual path to your CSV
image_folder = "new_imag/"  # Replace with the actual path to your images
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]

# Load the CSV data
robot_data = load_csv(csv_path)

# Load T_gripper_to_marker (constant transformation)
T_gripper_to_marker = np.load('T_gripper_to_marker_new.npy')  # Load from file or define here

# Initial T_base_to_camera (to be optimized)
T_base_to_camera = np.load('T_base_to_camera_new.npy')  # Load initial estimate

# Lists to store data for optimization
T_base_to_gripper_list = []
image_points_list = []
object_points_list = []

# Define the marker model points (3D points in marker coordinate system)
markerLength = 0.1  # Marker size in meters
half_size = markerLength / 2.0
marker_model_points = np.array([
    [-half_size,  half_size, 0],
    [ half_size,  half_size, 0],
    [ half_size, -half_size, 0],
    [-half_size, -half_size, 0]
])

# Loop through images and collect data
for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(corners) > 0:
        try:
            # Match the CSV entry and extract transformation
            csv_row = match_csv_row(os.path.basename(image_path), robot_data)
            R_bg, t_bg = extract_robot_transformation(csv_row)

            # Build T_base_to_gripper
            T_base_to_gripper = np.eye(4)
            T_base_to_gripper[:3, :3] = R_bg
            T_base_to_gripper[:3, 3] = t_bg

            # For each detected marker
            for i in range(len(corners)):
                # Collect data
                T_base_to_gripper_list.append(T_base_to_gripper)
                image_points_list.append(corners[i][0])  # Detected 2D marker corners
                object_points_list.append(marker_model_points)  # Corresponding 3D marker points

        except ValueError as e:
            print(f"Error: {e}. Skipping image {image_path}")
    else:
        print(f"Warning: ArUco marker not detected in image {image_path}. Skipping.")

# Optimization function to minimize the reprojection error
def optimize_T_base_to_camera(T_base_to_camera, T_base_to_gripper_list, T_gripper_to_marker, object_points_list, image_points_list):

    def reprojection_error(params):
        # Construct the optimized T_base_to_camera
        tvec = params[:3]
        T_base_to_camera_opt = np.eye(4)
        T_base_to_camera_opt[:3, :3] = T_base_to_camera[:3, :3]
        T_base_to_camera_opt[:3, 3] = tvec

        error = []

        # Precompute inverse of T_base_to_camera
        T_camera_to_base_opt = np.linalg.inv(T_base_to_camera_opt)

        for T_base_to_gripper, obj_pts, img_pts in zip(T_base_to_gripper_list, object_points_list, image_points_list):
            # Compute T_base_to_marker
            T_base_to_marker = T_base_to_gripper @ T_gripper_to_marker

            # Compute T_camera_to_marker
            T_camera_to_marker = T_camera_to_base_opt @ T_base_to_marker

            # Project the 3D marker points to 2D image points
            rvec_cm, _ = cv2.Rodrigues(T_camera_to_marker[:3, :3])
            tvec_cm = T_camera_to_marker[:3, 3]

            projected_points, _ = cv2.projectPoints(
                obj_pts, rvec_cm, tvec_cm, camera_matrix, dist_coeffs
            )

            # Compute reprojection error
            error.append((projected_points.squeeze() - img_pts).ravel())

        return np.concatenate(error)

    # Initial parameters (rvec and tvec from initial T_base_to_camera)
    R_base_to_camera = T_base_to_camera[:3, :3]
    t_init = T_base_to_camera[:3, 3]
    initial_params = t_init

    # Optimize using Levenberg-Marquardt
    result = least_squares(reprojection_error, initial_params, method="lm")

    # Return optimized T_base_to_camera
    optimized_T = np.eye(4)
    optimized_T[:3, :3] = R_base_to_camera
    optimized_T[:3, 3] = result.x[:3]
    return optimized_T

# Optimize T_base_to_camera
T_base_to_camera_optimized = optimize_T_base_to_camera(T_base_to_camera, T_base_to_gripper_list, T_gripper_to_marker, object_points_list, image_points_list)

# Output the optimized transformation matrix
print("Optimized T_base->T_camera (SE(3)):\n", T_base_to_camera_optimized)

# Save the optimized transformation matrix
np.save("T_base_to_camera_optimized.npy", T_base_to_camera_optimized)
