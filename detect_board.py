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


# Draw holes on the image
def draw_holes_on_image(image, hole_positions, T_board_to_camera, camera_matrix, dist_coeffs):
    """
    Draw large dots on the image at the projected hole positions.
    """
    for hole in hole_positions:
        # Convert hole position to the camera frame
        hole_homogeneous = np.array([*hole, 0, 1])  # Assuming z = 0 in the board frame
        hole_in_camera = T_board_to_camera @ hole_homogeneous

        # Project the 3D point to the image plane
        projected_points, _ = cv2.projectPoints(
            hole_in_camera[:3].reshape(1, 3),  # Convert to (1, 3)
            np.zeros(3),  # Rotation vector (identity rotation)
            np.zeros(3),  # Translation vector (no additional translation)
            camera_matrix,
            dist_coeffs,
        )

        # Draw a large dot at the projected point
        center = tuple(map(int, projected_points[0].ravel()))
        cv2.circle(image, center, 10, (0, 0, 255), -1)  # Red dots


# Draw the X and Y axes of the board on the image
def draw_board_axes(image, T_board_to_camera, camera_matrix, dist_coeffs):
    """
    Draw the X and Y axes of the board on the image, with the origin at the ArUco marker with the lower ID.
    """
    # Define the axes in the board's local frame (meters)
    axis_points = np.array([
        [0, 0, 0],      # Origin
        [0.1, 0, 0],    # X-axis (10 cm along X)
        [0, 0.1, 0],    # Y-axis (10 cm along Y)
    ], dtype=np.float32)

    # Transform the axis points to the camera frame
    axis_points_camera = np.dot(T_board_to_camera[:3, :3], axis_points.T).T + T_board_to_camera[:3, 3]

    # Project the 3D axis points to the image plane
    projected_points, _ = cv2.projectPoints(
        axis_points_camera,
        np.zeros(3),  # Identity rotation
        np.zeros(3),  # No additional translation
        camera_matrix,
        dist_coeffs,
    )

    # Convert to integer pixel positions
    origin = tuple(map(int, projected_points[0].ravel()))
    x_axis = tuple(map(int, projected_points[1].ravel()))
    y_axis = tuple(map(int, projected_points[2].ravel()))

    # Draw the axes on the image
    cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 2, tipLength=0.2)  # Red for X-axis
    cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 2, tipLength=0.2)  # Green for Y-axis
    cv2.putText(image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# Transform hole positions to the robot's base frame
def transform_hole_positions_to_base(hole_positions, T_base_to_board):
    """
    Transform hole positions from the board's local frame to the robot's base frame.
    """
    transformed_positions = []
    for hole in hole_positions:
        # Convert to homogeneous coordinates
        hole_homogeneous = np.array([*hole, 0, 1])  # Assuming z = 0 in the board frame
        hole_in_base = T_base_to_board @ hole_homogeneous
        transformed_positions.append(hole_in_base[:3])  # Drop the homogeneous coordinate
    return transformed_positions


# Main script
if __name__ == "__main__":
    # Load the transformation matrix (SE(3)) and camera parameters
    T_base_to_camera = np.load("T_base_to_camera.npy")  # Replace with your file path
    data = np.load('calibration_data.npz')
    camera_matrix = data['K']
    dist_coeffs = data['dist']

    # Load the image
    image_path = "detection_images/img01.png"  # Replace with the actual image path
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(corners) > 0:
        T_base_to_board = None

        # Variables to store the lower-ID marker's pose
        rvecs_lower, tvecs_lower = None, None

        for i, corner in enumerate(corners):
            # Estimate pose of each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corner], markerLength=0.036, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

            # Draw the marker and its axes
            cv2.aruco.drawDetectedMarkers(img, [corner], ids[i])
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)

            # Check if this is the lower-ID marker
            if ids[i] == 1:  # Replace with the correct ID of the reference marker
                rvecs_lower, tvecs_lower = rvecs[0], tvecs[0]

        if rvecs_lower is not None and tvecs_lower is not None:
            # Convert rvec to rotation matrix
            R_camera_to_board, _ = cv2.Rodrigues(rvecs_lower)

            # Create SE(3) for the board in the camera frame
            T_camera_to_board = np.eye(4)
            T_camera_to_board[:3, :3] = R_camera_to_board
            T_camera_to_board[:3, 3] = tvecs_lower.ravel()

            # Transform to the world (robot base) frame
            T_base_to_board = T_base_to_camera @ T_camera_to_board

            # Load hole positions and transform them
            hole_positions = load_csv("positions_plate_01-02.csv")
            transformed_holes = transform_hole_positions_to_base(hole_positions, T_base_to_board)

            # Print transformed positions
            print("Transformed Hole Positions in Robot Base Frame:")
            for pos in transformed_holes:
                print(pos)

            # Draw holes and axes for visualization
            draw_holes_on_image(img, hole_positions, T_camera_to_board, camera_matrix, dist_coeffs)
            draw_board_axes(img, T_camera_to_board, camera_matrix, dist_coeffs)

            # Show the result
            while True:
                cv2.imshow("Detected ArUco Markers with Board Axes and Holes", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
            cv2.destroyAllWindows()
        else:
            print("Lower-ID marker not detected.")
    else:
        print("No ArUco markers detected.")
