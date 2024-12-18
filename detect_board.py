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
        self.T_camera_to_board_lower = []
        self.T_camera_to_board_higher = []

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

    def transform_hole_positions_to_base_average(self, hole_positions, T_camera_to_high, T_camera_to_low, board_id):
        """
        Calculate hole positions relative to both markers and average them.
        """
        T_camera_to_high, T_camera_to_low = T_camera_to_high[board_id], T_camera_to_low[board_id]
        hole_positions_high = []
        hole_positions_low = []
        hole_positions_higher_z = []

        for hole in hole_positions:
            # Convert hole to homogeneous coordinates
            hole_homogeneous = np.array([*hole, 0, 1])

            # Transform relative to the higher-ID marker
            hole_in_high = self.T_base_to_camera @ T_camera_to_high @ (hole_homogeneous - np.array([0.18, 0.14, 0, 0]))
            hole_positions_high.append(hole_in_high[:3])

            # Transform relative to the lower-ID marker
            hole_in_low = self.T_base_to_camera @ T_camera_to_low @ hole_homogeneous
            hole_positions_low.append(hole_in_low[:3])
            
            if hole_in_high[2] > hole_in_low[2]:
                hole_positions_higher_z.append(hole_in_high[:3])
            else:
                hole_positions_higher_z.append(hole_in_low[:3])
        # # Average the positions
        # hole_positions_avg = [(np.array(high) + np.array(low)) / 2.0 for high, low in zip(hole_positions_high, hole_positions_low)]

        return hole_positions_higher_z

    def detect_and_transform_holes(self, image_path):
        """
        Detect ArUco markers, estimate the transformation matrix, and return the transformed hole positions.
        """
        # Load the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # h, w = gray.shape[:2]
        # new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        # gray = cv2.undistort(gray, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        # Initialize the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()

        # Detect markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        refined_corners_list = []

        for i in range(len(corners)):
            refined_corners = cv2.cornerSubPix(
                gray,
                corners[i],
                winSize=(5,5),
                zeroZone=(-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001),
            )
            refined_corners_list.append(refined_corners)

        corners = refined_corners_list

        if ids is not None and len(corners) >= 2:  # Ensure we detect at least one board
            dirky = []
            tfs = []
            ids = ids.flatten()

            # Sort markers by ID
            sorted_indices = np.argsort(ids)
            ids = ids[sorted_indices]
            corners = [corners[i] for i in sorted_indices]

            # Pair markers sequentially (n, n+1)
            boards = []
            # print(ids)
            for i in range(0, len(ids) - 1, 2):
                if ids[i + 1] == ids[i] + 1:
                    boards.append(((ids[i], corners[i]), (ids[i + 1], corners[i + 1])))

            # Process each board
            for board_id, (marker_low, marker_high) in enumerate(boards):
                (id_low, corner_low), (id_high, corner_high) = marker_low, marker_high

                # Estimate pose of each marker
                rvec_low, tvec_low, _ = aruco.estimatePoseSingleMarkers([corner_low], markerLength=0.036, cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs)
                rvec_high, tvec_high, _ = aruco.estimatePoseSingleMarkers([corner_high], markerLength=0.036, cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs)

                # Convert rvec to rotation matrices
                R_low, _ = cv2.Rodrigues(rvec_low[0])
                R_high, _ = cv2.Rodrigues(rvec_high[0])

                self.T_camera_to_board_lower.append(self.get_T_camera_to_board(rvec_low, tvec_low)) 
                self.T_camera_to_board_higher.append(self.get_T_camera_to_board(rvec_high, tvec_high))

                # Load hole positions
                hole_positions = self.load_csv(f"positions_plate_0{id_low}-0{id_high}.csv")  # Adjust for board-specific CSV

                # Calculate hole positions relative to both markers and their average
                hole_positions_avg = self.transform_hole_positions_to_base_average(hole_positions, self.T_camera_to_board_higher, self.T_camera_to_board_lower, board_id)
                dirky.append(hole_positions_avg)
                tfs.append(self.T_camera_to_board_lower)
            return dirky, img, tfs
        else:
            raise ValueError("No ArUco markers detected.")

    def get_T_camera_to_board(self, rvec, tvec):

        if rvec is not None and tvec is not None:
            # Convert rvec to rotation matrix
            R_camera_to_board, _ = cv2.Rodrigues(rvec)

            # Create SE(3) for the board in the camera frame
            T_camera_to_board = np.eye(4)
            T_camera_to_board[:3, :3] = R_camera_to_board
            T_camera_to_board[:3, 3] = tvec.ravel()

            return T_camera_to_board
        else:
            raise ValueError("marker not detected.")
        return None

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

    def refine_hole_positions(self, hole_positions_list):
        """
        Refine hole positions using multiple transformations and corresponding positions.
        
        Args:
            hole_positions_list (list of np.ndarray): List of hole position arrays (one array per image).
            transforms_list (list of np.ndarray): List of T_camera_to_board transformations (one per image).

        Returns:
            refined_hole_positions (np.ndarray): Refined hole positions with improved z-components.
        """

        num_holes = hole_positions_list.shape[1]  
        num_images = hole_positions_list.shape[0]

        # Collect z-values for each hole across all images
        z_values_per_hole = [[] for _ in range(num_holes)]
        refined_hole_positions = []

        for i in range(num_images):
            hole_positions = hole_positions_list[i]

        for j, hole in enumerate(hole_positions): 
            hole_in_camera = hole
            # Collect the z-component
            z_values_per_hole[j].append(hole_in_camera[2])

        # Process each hole's z-values
        for j in range(num_holes):
            z_values = np.array(z_values_per_hole[j])
            print(np.min(z_values), np.max(z_values))
            # Refine the z-component using robust statistics (e.g., median)
            refined_z = np.median(z_values)

            # Use the first image's x, y as reference (assuming x, y are consistent across images)
            refined_hole = np.array([hole_positions_list[0][j][0], hole_positions_list[0][j][1], refined_z])
            refined_hole_positions.append(refined_hole)

        return np.array(refined_hole_positions)