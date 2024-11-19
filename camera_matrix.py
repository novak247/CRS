import cv2
import numpy as np
import glob

# Nastavení šachovnice
checkerboard_size = (7, 4)  # Vnitřní rohy šachovnice (např. 7x7)
square_size = 0.05  # Velikost čtverce v metrech

# Příprava 3D bodů šachovnice (0,0,0), (1,0,0), (2,0,0), ..., (6,6,0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Přizpůsobení velikosti čtverce

# Seznamy pro uložení 3D bodů a 2D bodů z obrázků
obj_points = []  # 3D body ve světové soustavě
img_points = []  # 2D body v obraze

# Načtení obrázků šachovnice
images = ['images/Image__2024-11-19__09-20-27.bmp', 'images/Image__2024-11-19__09-20-59.bmp', 'images/Image__2024-11-19__09-21-07.bmp']  # Upravte cesty k obrázkům

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Najděte rohy šachovnice
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        # Kreslení rohů
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Rohy šachovnice nebyly nalezeny v obrázku {fname}")

cv2.destroyAllWindows()

# Kalibrace kamery
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

if ret:
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
else:
    print("Kalibrace se nezdařila.")
