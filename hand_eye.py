import cv2
import cv2.aruco as aruco
import numpy as np
import re
import os
import math

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
camera_matrix = np.load("cameraMatrix.npy")
camera_matrix[0,2] = 960
camera_matrix[1,2] = 600
# camera_matrix = data["K"]
# dist_coeffs = data['dist']
dist_coeffs = np.load("distCoeffs.npy")

# Paths to images and CSV
csv_path = "new_transformations.csv"  # Replace with the actual path to your CSV
image_folder = "new_imag/"  # Replace with the actual path to your images
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]

# Load the CSV data
robot_data = load_csv(csv_path)

camera_rotations = []
camera_translations = []
robot_rotations = []
robot_translations = []
not_detected = 0

for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(corners) > 0:
        for i in range(len(corners)):
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[i]], markerLength=0.1, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

            rotation_matrix, _ = cv2.Rodrigues(rvecs[0])

            try:
                # Match the CSV entry and extract transformation
                csv_row = match_csv_row(os.path.basename(image_path), robot_data)
                R_bg, t_bg = extract_robot_transformation(csv_row)

                # Combine transformations: T_base_to_gripper * T_gripper_to_marker
                T_bg = np.eye(4)
                T_bg[:3, :3] = R_bg
                T_bg[:3, 3] = t_bg

                # Store the results for each marker
                camera_rotations.append(rotation_matrix)
                camera_translations.append(tvecs[0].ravel())
                robot_rotations.append(T_bg[:3, :3])
                robot_translations.append(T_bg[:3, 3])

                print(f"Detected marker {ids[i]} with transformation:")
                print(f"Rotation Matrix:\n{rotation_matrix}")
                print(f"Translation Vector: {tvecs[0].ravel()}")
            except ValueError as e:
                print(f"Error: {e}. Skipping image {image_path}")
    else:
        print(f"Warning: ArUco marker not detected in image {image_path}. Skipping.")
        not_detected += 1


R_gripper_to_marker, t_gripper_to_marker, R_base_to_camera, t_base_to_camera = cv2.calibrateRobotWorldHandEye(
    robot_rotations, robot_translations,
    camera_rotations, camera_translations,
    method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH  # You can choose other methods as well
)

# Create SE(3) matrix for T_base->T_camera
T_base_to_camera = np.eye(4)
T_base_to_camera[:3, :3] = R_base_to_camera
T_base_to_camera[:3, 3] = t_base_to_camera.ravel()
np.save("T_base_to_camera_new.npy", T_base_to_camera)

T_camera_to_base = np.eye(4)
T_camera_to_base[:3, :3] = R_base_to_camera.T
T_camera_to_base[:3, 3] = -R_base_to_camera.T @ T_base_to_camera[:3, 3]
np.save("T_camera_to_base_new.npy", T_camera_to_base)

T_gripper_to_marker = np.eye(4)
T_gripper_to_marker[:3, :3] = R_gripper_to_marker
T_gripper_to_marker[:3, 3] = t_gripper_to_marker.ravel()
np.save("T_gripper_to_marker_new.npy", T_gripper_to_marker)

# Output the transformation matrix
print("Transformation matrix T_base->T_camera (SE(3)):\n", T_base_to_camera)
print("Transformation matrix T_gripper->T_marker (SE(3)):\n", T_gripper_to_marker)
