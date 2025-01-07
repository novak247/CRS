import cv2
import cv2.aruco as aruco
import numpy as np

# Load the transformation matrix (SE(3)) and camera parameters
T_base_to_camera = np.load("T_base_to_camera.npy")  # Replace with your file path
data = np.load('calibration_data.npz')
camera_matrix = data['K']
dist_coeffs = data['dist']

# Load the image
image_path = "board_imgs/hImage5.png"  # Replace with the actual image path
img = cv2.imread(image_path)
img = cv2.resize(img, (960, 600))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Detect markers
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None and len(corners) > 0:
    for i, corner in enumerate(corners):
        # Estimate pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corner], markerLength=0.036, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

        # Draw the marker and its axes
        cv2.aruco.drawDetectedMarkers(img, [corner], ids[i])
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)

        # Translation vector of the marker in the camera frame
        t_camera = tvecs[0].ravel()  # [x, y, z] in camera frame

        # Convert rvec to rotation matrix
        R_camera, _ = cv2.Rodrigues(rvecs[0])
        # Create SE(3) for the marker in the camera frame
        T_camera_to_marker = np.eye(4)
        T_camera_to_marker[:3, :3] = R_camera
        T_camera_to_marker[:3, 3] = t_camera

        # Transform to the world (robot base) frame
        T_base_to_marker = T_base_to_camera @ T_camera_to_marker
        t_base = T_base_to_marker[:3, 3]  # Extract translation in the base frame

    

        # Print the coordinates in the base frame
        print(f"Marker ID {ids[i]} in base frame:")
        print(f"  Origin: {t_base}")
    
    # Show the image with markers and axes drawn
    while True:
        
        cv2.imshow("Detected ArUco Markers", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cv2.destroyAllWindows()
else:
    print("No ArUco markers detected.")
