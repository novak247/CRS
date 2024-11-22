import cv2
import numpy as np
import time
from basler_camera import BaslerCamera

# Chessboard settings
checkerboard_size = (15, 10)  # Inner corners per chessboard row and column

# Initialize camera
camera = BaslerCamera()
camera.connect_by_name("camera-crs97")
camera.open()
camera.set_parameters()
camera.start()

image_count = 0
max_images = 20  # Number of images to capture
capture_interval = 5  # Time in seconds between captures

print("Starting image capture...")
time.sleep(10)

try:
    while image_count < max_images:
        img = camera.grab_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            image_count += 1
            filename = f'chessboard_{image_count}.png'
            cv2.imwrite(filename, img)
            print(f"Corners detected. Saved image {filename}")
        else:
            print("No corners detected.")

        time.sleep(capture_interval)

except KeyboardInterrupt:
    print("Image capture interrupted.")

finally:
    camera.stop()
    camera.close()
    print("Camera closed.")
