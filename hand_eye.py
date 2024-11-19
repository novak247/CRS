import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import re

# Funkce pro extrakci translací z názvů souborů
def extract_translation_from_filename(filename):
    match = re.search(r'X([-+]?[0-9]*\.?[0-9]+)Y([-+]?[0-9]*\.?[0-9]+)Z([-+]?[0-9]*\.?[0-9]+)', filename)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))
        return np.array([x, y, z])
    else:
        raise ValueError(f"Nelze extrahovat translaci z názvu souboru: {filename}")

# Přidání vektoru translace od gripperu ke středu markeru
gripper_to_marker_translation = np.array([0.13, -0.03, -0.15])  # V mm (převod na metry, pokud je potřeba)

# Parametry kamery (nahraďte skutečnými hodnotami z vaší kalibrace)
camera_matrix = np.array([[1.23781535e+04, 0, 8.73297003e+02],
                          [0, 1.23969122e+04, 5.59542074e+02],
                          [0, 0, 1]])
dist_coeffs = np.array([-2.45263371e+00,  9.80828533e+01,  2.02541409e-02,  1.09737521e-02,   6.12911283e-01])  # Nebo použijte skutečné zkreslovací koeficienty

# Cesty k obrázkům s ArUco markery
image_paths = ['images/X0.3305139639208873Y-0.10039639690878899Z0.41029718670543414.png', 'images/X0.3798957533959076Y0.09988492405723545Z0.41049852405412945.png',
'images/X0.3799044841576267Y-0.0015396300550540522Z0.4105613205042755.png', 'images/X0.3800826965933747Y0.09960222869450001Z0.3097266742190109.png',
'images/X0.3801069416216306Y-0.0007343977366757379Z0.4103652212283435.png', 'images/X0.3801693779762797Y-0.10139472508223313Z0.41046033423400363.png',
'images/X0.3801950644047894Y-0.0006270700297724952Z0.3097778472970399.png', 'images/X0.32979939292788796Y0.10067749524885185Z0.4100805803328725.png',
'images/X0.38036924267966765Y-0.10089144742506458Z0.3097248936709808.png']  # Nahraďte cestou ke složce s obrázky

# Seznamy pro ukládání rotačních a translačních vektorů z kamery (T_CT)
camera_rotations = []
camera_translations = []
robot_rotations = []
robot_translations = []

for image_path in image_paths:
    # Načtení obrázku
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detekce markerů
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Najdeme největší marker podle plochy
        largest_marker_index = np.argmax([cv2.contourArea(corner) for corner in corners])

        # Odhad pozice a orientace pro největší marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[largest_marker_index]], markerLength=0.08, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

        rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
        camera_rotations.append(rotation_matrix)
        camera_translations.append(tvecs[0].ravel())

        # Extrakce translace z názvu souboru pro T_RG a přičtení vektoru od gripperu ke středu markeru
        translation = extract_translation_from_filename(image_path) + gripper_to_marker_translation
        print(translation)
        robot_rotations.append(np.array([[0, 1, 0], [-1, 0, 0], [0, 1, 0]]))  
        robot_translations.append(translation)
    else:
        print(f"ArUco marker nebyl detekován v obrázku {image_path}")

# Kalibrace oko-ruka
hand_eye_rotation, hand_eye_translation = cv2.calibrateHandEye(
    robot_rotations, robot_translations,
    camera_rotations, camera_translations,
    method=cv2.CALIB_HAND_EYE_TSAI  # Můžete zkusit jiné metody: PARK, HORAUD, ANDREFF
)

# Vytvoření SE(3) matice 4x4 pro T_base->T_camera
T_base_to_camera = np.eye(4)
T_base_to_camera[:3, :3] = hand_eye_rotation
T_base_to_camera[:3, 3] = hand_eye_translation.ravel()

# Výstup výsledné transformace
print("Transformační matice T_base->T_camera (SE(3)):\n", T_base_to_camera)
