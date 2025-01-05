import cv2
import cv2.aruco as aruco
import numpy as np
import csv


def load_csv(file_path):
    """Load the CSV file containing hole positions on the board."""
    with open(file_path, 'r') as file:
        data = file.readlines()
    hole_positions_mm = [list(map(float, line.strip().split(','))) for line in data[1:]]
    hole_positions_m = [[x / 1000.0, y / 1000.0] for x, y in hole_positions_mm]
    return hole_positions_m


def calculate_hole_positions_relative(hole_positions, T_camera_to_low, T_camera_to_high):
    """Calculate hole positions relative to both markers and average them."""
    hole_positions_marker1 = []
    hole_positions_marker2 = []

    for hole in hole_positions:
        hole_homogeneous = np.array([*hole, 0, 1])
        hole_in_marker1 = T_camera_to_low @ hole_homogeneous
        hole_positions_marker1.append(hole_in_marker1[:3])
        hole_in_marker2 = T_camera_to_high @ (hole_homogeneous - np.array([0.18, 0.14, 0, 0]))
        hole_positions_marker2.append(hole_in_marker2[:3])

    hole_positions_avg = [(np.array(m1) + np.array(m2)) / 2.0 for m1, m2 in zip(hole_positions_marker1, hole_positions_marker2)]

    return hole_positions_marker1, hole_positions_marker2, hole_positions_avg


def draw_hole_positions(image, hole_positions_camera, camera_matrix, dist_coeffs, color, label):
    """Draw large dots for hole positions in the camera frame on the image."""
    for idx, hole_camera in enumerate(hole_positions_camera):
        projected_points, _ = cv2.projectPoints(
            np.array(hole_camera).reshape(1, 3),
            np.zeros(3),  # No rotation
            np.zeros(3),  # No translation
            camera_matrix,
            dist_coeffs,
        )
        center = tuple(map(int, projected_points[0].ravel()))
        cv2.circle(image, center, 6, color, -1)
        cv2.putText(image, f"{label}{idx}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# Main script
if __name__ == "__main__":
    # Load the transformation matrix (SE(3)) and camera parameters
    T_base_to_camera = np.load("T_base_to_camera_new.npy")  # Replace with your file path
    data = np.load('calibration_data.npz')
    camera_matrix = data['K']
    dist_coeffs = data['dist']

    # Load the image
    image_path = "detection_images/boards1.png"  # Replace with the actual image path
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(corners) >= 2:  # Ensure we detect at least one board
        ids = ids.flatten()

        # Sort markers by ID
        sorted_indices = np.argsort(ids)
        ids = ids[sorted_indices]
        corners = [corners[i] for i in sorted_indices]

        # Pair markers sequentially (n, n+1)
        boards = []
        print(ids)
        for i in range(0, len(ids) - 1, 2):
            if ids[i + 1] == ids[i] + 1:
                boards.append(((ids[i], corners[i]), (ids[i + 1], corners[i + 1])))

        # Process each board
        for board_id, (marker_low, marker_high) in enumerate(boards):
            (id_low, corner_low), (id_high, corner_high) = marker_low, marker_high


            # Estimate pose of each marker
            rvec_low, tvec_low, _ = aruco.estimatePoseSingleMarkers([corner_low], markerLength=0.036, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
            rvec_high, tvec_high, _ = aruco.estimatePoseSingleMarkers([corner_high], markerLength=0.036, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

            # Convert rvec to rotation matrices
            R_low, _ = cv2.Rodrigues(rvec_low[0])
            R_high, _ = cv2.Rodrigues(rvec_high[0])

            # Create SE(3) transformation matrices
            T_camera_to_low = np.eye(4)
            T_camera_to_low[:3, :3] = R_low
            T_camera_to_low[:3, 3] = tvec_low[0].ravel()

            T_camera_to_high = np.eye(4)
            T_camera_to_high[:3, :3] = R_high
            T_camera_to_high[:3, 3] = tvec_high[0].ravel()

            # Load hole positions
            hole_positions = load_csv(f"positions_plate_0{id_low}-0{id_high}.csv")  # Adjust for board-specific CSV

            # Calculate hole positions relative to both markers and their average
            holes_low, holes_high, holes_avg = calculate_hole_positions_relative(hole_positions, T_camera_to_low, T_camera_to_high)

            # Draw the holes for visualization
            draw_hole_positions(img, holes_low, camera_matrix, dist_coeffs, (255, 0, 0), f"B{board_id}L")  # Blue for low-ID marker
            draw_hole_positions(img, holes_high, camera_matrix, dist_coeffs, (0, 255, 0), f"B{board_id}H")  # Green for high-ID marker
            draw_hole_positions(img, holes_avg, camera_matrix, dist_coeffs, (0, 0, 255), f"B{board_id}A")  # Red for average
        # Show the result
        while True:
            cv2.imshow("Detected ArUco Markers with Hole Positions", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        cv2.destroyAllWindows()
    else:
        print("Not enough ArUco markers detected.")
