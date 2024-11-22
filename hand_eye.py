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

# Function to translate CSV offsets to approximate image values
def translate_csv_to_image_values(csv_x, csv_y, csv_z):
    # Map x offsets to approximate image values
    image_x = 0.38 + csv_x

    # y values are assumed to match directly
    image_y = csv_y

    # z value in the image is approximately 0.26 + csv_z
    image_z = 0.26 + csv_z

    return image_x, image_y, image_z

# Updated match_csv_row function
def match_csv_row(filename, csv_data):
    match = re.search(
        r'X([-+]?[0-9]*\.?[0-9]+)Y([-+]?[0-9]*\.?[0-9]+)Z([-+]?[0-9]*\.?[0-9]+)_J5([-+]?[0-9]*\.?[0-9]+)_J6([-+]?[0-9]*\.?[0-9]+)',
        filename
    )
    if match:
        image_x = float(match.group(1))
        image_y = float(match.group(2))
        image_z = float(match.group(3))
        joint_5 = match.group(4)
        joint_6 = match.group(5)

        for row in csv_data:
            csv_x, csv_y, csv_z = float(row["x"]), float(row["y"]), float(row["z"])
            approx_image_x, approx_image_y, approx_image_z = translate_csv_to_image_values(csv_x, csv_y, csv_z)

            # Check for match within a tolerance
            if (
                abs(image_x - approx_image_x) < 0.03 and  # Adjust tolerance as needed
                abs(image_y - approx_image_y) < 0.03 and
                abs(image_z - approx_image_z) < 0.03 and
                row["joint_5"] == joint_5 and
                row["joint_6"] == joint_6
            ):
                return row

    raise ValueError(f"Cannot find matching row in CSV for filename: {filename}")


# Transformation from gripper to marker center (rotation + translation)
gripper_to_marker_translation = np.array([0.0, 0.0, -0.10])  # Translation in meters
# Define the inverse of the combined rotation matrix
theta_x1 = math.radians(180)  # First rotation around X-axis
theta_z = math.radians(-90)   # Rotation around Z-axis
theta_y = math.radians(-90)  # Second rotation around X-axis

# Rotation matrix for 180° around X-axis
R_x1 = np.array([
    [1,  0,           0],
    [0,  math.cos(theta_x1), -math.sin(theta_x1)],
    [0,  math.sin(theta_x1),  math.cos(theta_x1)]
])

# Rotation matrix for -90° around Z-axis
R_z = np.array([
    [math.cos(theta_z), -math.sin(theta_z), 0],
    [math.sin(theta_z),  math.cos(theta_z), 0],
    [0,                 0,                 1]
])

R_y = np.array([
    [math.cos(theta_y),  0, math.sin(theta_y)],
    [0,                 1, 0],
    [-math.sin(theta_y), 0, math.cos(theta_y)]
])
R_combined = R_y @ R_z @ R_x1

# Combine into a 4x4 transformation matrix
T_gripper_to_marker = np.eye(4)
T_gripper_to_marker[:3, :3] = R_combined
T_gripper_to_marker[:3, 3] = gripper_to_marker_translation


# Camera parameters (replace with actual values)
data = np.load('calibration_data.npz')
camera_matrix = data['K']
dist_coeffs = data['dist']

# Paths to images and CSV
csv_path = "transformations.csv"  # Replace with the actual path to your CSV
image_folder = "images/"  # Replace with the actual path to your images
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

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(corners) > 0:
        largest_marker_index = np.argmax([cv2.contourArea(corner) for corner in corners])
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[largest_marker_index]], markerLength=0.06, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

        rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
        

        try:
            
            # Match the CSV entry and extract transformation
            csv_row = match_csv_row(os.path.basename(image_path), robot_data)
            R_bg, t_bg = extract_robot_transformation(csv_row)

            # Combine transformations: T_base_to_gripper * T_gripper_to_marker
            T_bg = np.eye(4)
            T_bg[:3, :3] = R_bg
            T_bg[:3, 3] = t_bg
            T_bm = np.dot(T_bg, T_gripper_to_marker)

            camera_rotations.append(rotation_matrix)
            camera_translations.append(tvecs[0].ravel())
            robot_rotations.append(T_bm[:3, :3])
            robot_translations.append(T_bm[:3, 3])
        except ValueError as e:
            print(f"Error: {e}. Skipping image {image_path}")
    else:
        print(f"Warning: ArUco marker not detected in image {image_path}. Skipping.")
        not_detected +=1


print("not detected: ", not_detected)
# Hand-eye calibration
print("R_gripper2base size:", len(robot_rotations))
print("t_gripper2base size:", len(robot_translations))
print("R_target2cam size:", len(camera_rotations))
print("t_target2cam size:", len(camera_translations))

R_gripper_to_camera, t_gripper_to_camera, R_base_to_camera, t_base_to_camera= cv2.calibrateRobotWorldHandEye(
    camera_rotations, camera_translations,
    robot_rotations, robot_translations,
    method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH  # You can choose other methods as well
)

# Create SE(3) matrix for T_base->T_camera
T_base_to_camera = np.eye(4)
T_base_to_camera[:3, :3] = R_base_to_camera
T_base_to_camera[:3, 3] = t_base_to_camera.ravel()
np.save("T_base_to_camera.npy", T_base_to_camera)
T_gripper_to_camera = np.eye(4)
T_gripper_to_camera[:3, :3] = R_gripper_to_camera
T_gripper_to_camera[:3, 3] = t_gripper_to_camera.ravel()
np.save("T_gripper_to_camera.npy", T_gripper_to_camera)
# Output the transformation matrix
print("Transformation matrix T_base->T_camera (SE(3)):\n", T_base_to_camera)
print("Transformation matrix T_gripper->T_camera (SE(3)):\n", T_gripper_to_camera)