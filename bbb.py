import cv2
import cv2.aruco as aruco
import numpy as np

# Function to detect ArUco markers
def detect_aruco_marker(image_path):
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary you are using
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)  # Example: 4x4 markers with 50 IDs
    parameters = aruco.DetectorParameters()

    # Detect ArUco markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Check if markers are detected
    if ids is not None:
        print("Detected ArUco marker(s) with IDs:", ids.flatten())

        # Draw the detected markers' outlines and their IDs
        img_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)

        # Display the image with the detected markers
        cv2.imshow("Detected ArUco Markers", img_markers)

        # Wait for the user to press a key to close the image window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No ArUco markers detected in the image.")

# Example usage
image_path = 'new_imag/X0.42985582265282535Y0.09974907524468807Z0.3604415649016036_J5-0.39269908169872414_J60.7853981633974483.png'  # Replace with the actual path to your image
detect_aruco_marker(image_path)
