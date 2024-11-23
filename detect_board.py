import cv2
import cv2.aruco as aruco
import numpy as np


class HoleTransformer:
    def __init__(self, T_base_to_camera_path, calibration_data_path):
        # Load transformation matrix and camera parameters
        self.T_base_to_camera = np.load(T_base_to_camera_path)
        calibration_data = np.load(calibration_data_path)
        self.camera_matrix = calibration_data['K']
        self.dist_coeffs = calibration_data['dist']
        self.marker_length = 0.036

    def load_csv(self, file_path):
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

    def transform_hole_positions_to_base(self, hole_positions, T_base_to_board):
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

    def detect_and_transform_holes(self, image_path):
        """
        Detect ArUco markers, estimate the transformation matrix, and return the transformed hole positions.
        """
        # Load the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Initialize the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()

        # Detect markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(corners) > 0:
            id_min = min(ids)  # Find the marker with the lowest ID
            rvecs_lower, tvecs_lower = None, None

            for i, corner in enumerate(corners):
                # Estimate pose of each marker
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    [corner], markerLength=self.marker_length, 
                    cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs
                )

                # Draw the marker and its axes
                cv2.aruco.drawDetectedMarkers(img, [corner], ids[i])
                cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], 0.05)

                # Check if this is the lower-ID marker
                if ids[i] == id_min[0]:
                    rvecs_lower, tvecs_lower = rvecs[0], tvecs[0]

            if rvecs_lower is not None and tvecs_lower is not None:
                # Convert rvec to rotation matrix
                R_camera_to_board, _ = cv2.Rodrigues(rvecs_lower)

                # Create SE(3) for the board in the camera frame
                T_camera_to_board = np.eye(4)
                T_camera_to_board[:3, :3] = R_camera_to_board
                T_camera_to_board[:3, 3] = tvecs_lower.ravel()

                # Transform to the world (robot base) frame
                T_base_to_board = self.T_base_to_camera @ T_camera_to_board

                # Load hole positions and transform them
                hole_positions = self.load_csv(f"positions_plate_0{id_min[0]}-0{id_min[0]+1}.csv")
                transformed_holes = self.transform_hole_positions_to_base(hole_positions, T_base_to_board)
                return transformed_holes, img, T_camera_to_board
            else:
                raise ValueError("Lower-ID marker not detected.")
        else:
            raise ValueError("No ArUco markers detected.")

    def visualize(self, img, hole_positions, T_camera_to_board):
        """
        Draw holes and board axes on the image for visualization.
        """
        self.draw_holes_on_image(img, hole_positions, T_camera_to_board)
        self.draw_board_axes(img, T_camera_to_board)

        while True:
            cv2.imshow("Detected ArUco Markers with Board Axes and Holes", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        cv2.destroyAllWindows()

    def draw_holes_on_image(self, image, hole_positions, T_board_to_camera):
        """
        Draw large dots on the image at the projected hole positions.
        """
        for hole in hole_positions:
            hole_homogeneous = np.array([*hole, 0, 1])
            hole_in_camera = T_board_to_camera @ hole_homogeneous

            projected_points, _ = cv2.projectPoints(
                hole_in_camera[:3].reshape(1, 3),
                np.zeros(3),
                np.zeros(3),
                self.camera_matrix,
                self.dist_coeffs,
            )
            center = tuple(map(int, projected_points[0].ravel()))
            cv2.circle(image, center, 10, (0, 0, 255), -1)

    def draw_board_axes(self, image, T_board_to_camera):
        """
        Draw the X and Y axes of the board on the image.
        """
        axis_points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0, 0.1, 0],
        ], dtype=np.float32)

        axis_points_camera = np.dot(T_board_to_camera[:3, :3], axis_points.T).T + T_board_to_camera[:3, 3]
        projected_points, _ = cv2.projectPoints(
            axis_points_camera,
            np.zeros(3),
            np.zeros(3),
            self.camera_matrix,
            self.dist_coeffs,
        )

        origin = tuple(map(int, projected_points[0].ravel()))
        x_axis = tuple(map(int, projected_points[1].ravel()))
        y_axis = tuple(map(int, projected_points[2].ravel()))

        cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 2, tipLength=0.2)
        cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
