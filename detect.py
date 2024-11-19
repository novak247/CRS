import cv2
import cv2.aruco as aruco
import numpy as np

# Parametry kamery (nahraďte skutečnými hodnotami z vaší kalibrace)
camera_matrix = np.array([[1.23781535e+04, 0, 8.73297003e+02],
                          [0, 1.23969122e+04, 5.59542074e+02],
                          [0, 0, 1]])
dist_coeffs = np.array([-2.45263371e+00,  9.80828533e+01,  2.02541409e-02,  1.09737521e-02,   6.12911283e-01])  # Nebo použijte skutečné zkreslovací koeficienty

# Načtení obrázku
image_path = 'images/X0.38011945093997723Y-0.2021122462273852Z0.3097052118707556.png'  # Nahraďte cestou k obrázku
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inicializace ArUco slovníku
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Detekce markerů
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
print(ids)
if ids is not None:
    # Find the largest marker by area
    areas = [cv2.contourArea(corner) for corner in corners]
    largest_marker_index = np.argmax(areas)
    # Draw the detected largest marker
    cv2.aruco.drawDetectedMarkers(img, [corners[largest_marker_index]], ids[largest_marker_index])

    # Estimate pose and draw axes for the largest marker
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        [corners[largest_marker_index]], markerLength=0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
    )
    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)

    # Display the resulting image
    while True:
        cv2.imshow('Detected Largest Marker with Axes', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cv2.destroyAllWindows()
else:
    print("ArUco marker nebyl detekován.")
