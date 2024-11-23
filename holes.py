import cv2
import cv2.aruco as aruco
import numpy as np
import csv


# Load the hole positions from CSV
def load_csv(file_path):
    """
    Load the CSV file containing hole positions on the board.
    Convert positions from millimeters to meters.
    """
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Skip the first line (IDs of ArUco markers) and read the hole positions
    hole_positions_mm = [list(map(float, line.strip().split(','))) for line in data[1:]]
    
    # Convert from mm to meters
    hole_positions_m = [[x / 1000.0, y / 1000.0] for x, y in hole_positions_mm]
    return hole_positions_m


# Calculate hole positions relative to the higher-ID marker
def calculate_hole_positions_relative(hole_positions, T_camera_to_high, T_camera_to_low):
    """
    Calculate hole positions relative to both markers and average them.
    """
    hole_positions_high = []
    hole_positions_low = []

    for hole in hole_positions:
        # Convert hole to homogeneous coordinates
        hole_homogeneous = np.array([*hole, 0, 1])

        # Transform relative to the higher-ID marker
        hole_in_high = T_camera_to_high @ (hole_homogeneous - np.array([0.18, 0.14, 0, 0]))
        hole_positions_high.append(hole_in_high[:3])

        # Transform relative to the lower-ID marker
        hole_in_low = T_camera_to_low @ hole_homogeneous
        hole_positions_low.append(hole_in_low[:3])

    # Average the positions
    hole_positions_avg = [(np.array(high) + np.array(low)) / 2.0 for high, low in zip(hole_positions_high, hole_positions_low)]

    return hole_positions_high, hole_positions_low, hole_positions_avg


# Draw holes on the image
def draw_hole_positions(image, hole_positions_camera, camera_matrix, dist_coeffs, color, label):
    """
    Draw large dots for hole positions in the camera frame on the image.
    """
    for idx, hole_camera in enumerate(hole_positions_camera):
        # Project the 3D point to the image plane
        projected_points, _ = cv2.projectPoints(
            np.array(hole_camera).reshape(1, 3),
            np.zeros(3),  # No rotation
            np.zeros(3),  # No translation
            camera_matrix,
            dist_coeffs,
        )

        # Draw the dot
        center = tuple(map(int, projected_points[0].ravel()))
        cv2.circle(image, center, 6, color, -1)

        # Add label
        cv2.putText(image, f"{label}{idx}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# Main script
if __name__ == "__main__":
    # Load the transformation matrix (SE(3)) and camera parameters
    T_base_to_camera = np.load("T_base_to_camera.npy")  # Replace with your file path
    data = np.load('calibration_data.npz')
    camera_matrix = data['K']
    dist_coeffs = data['dist']

    # Load the image
    image_path = "imgs/holeImage.png"  # Replace with the actual image path
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(corners) >= 2:
        # Variables to store transformations
        T_camera_to_high = None
        T_camera_to_low = None

        for i, corner in enumerate(corners):
            # Estimate pose of each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corner], markerLength=0.036, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

            # Draw the marker and its axes
            cv2.aruco.drawDetectedMarkers(img, [corner], ids[i])
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)

            # Convert rvec to rotation matrix
            R_camera_to_marker, _ = cv2.Rodrigues(rvecs[0])

            # Create SE(3) for the marker in the camera frame
            T_camera_to_marker = np.eye(4)
            T_camera_to_marker[:3, :3] = R_camera_to_marker
            T_camera_to_marker[:3, 3] = tvecs[0].ravel()

            # Assign the transformations based on marker ID
            if ids[i] == np.min(ids):  # Lower-ID marker
                T_camera_to_low = T_camera_to_marker
            elif ids[i] == np.max(ids):  # Higher-ID marker
                T_camera_to_high = T_camera_to_marker

        if T_camera_to_high is not None and T_camera_to_low is not None:
            # Load hole positions
            hole_positions = load_csv("positions_plate_01-02.csv")

            # Calculate hole positions relative to both markers and their average
            holes_high, holes_low, holes_avg = calculate_hole_positions_relative(hole_positions, T_camera_to_high, T_camera_to_low)

            # Draw the holes for visualization
            draw_hole_positions(img, holes_high, camera_matrix, dist_coeffs, (0, 255, 0), "H")  # Green for high-ID marker
            draw_hole_positions(img, holes_low, camera_matrix, dist_coeffs, (255, 0, 0), "L")  # Blue for low-ID marker
            draw_hole_positions(img, holes_avg, camera_matrix, dist_coeffs, (0, 0, 255), "A")  # Red for average

            # Show the result
            while True:
                cv2.imshow("Detected ArUco Markers with Hole Positions", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
            cv2.destroyAllWindows()
        else:
            print("One or both reference markers not detected.")
    else:
        print("Not enough ArUco markers detected.")
