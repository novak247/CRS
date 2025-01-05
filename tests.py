import numpy as np
import math

# def normalize_angle(angle_radians):
#     print("---------", np.degrees(angle_radians))
#     # Convert the angle to the range [0, 2*pi)
#     angle_radians = angle_radians % (2 * math.pi)
    
#     # Convert the angle to degrees
#     angle_degrees = math.degrees(angle_radians)
    
#     # Normalize to the interval [0, 90]
#     normalized_angle = angle_degrees % 90
#     if normalized_angle > 45:
#       normalized_angle = normalized_angle - 90
#     return np.radians(normalized_angle)

def normalize_angle(angle_radians):
    print("---------", np.degrees(angle_radians))

    while (angle_radians < -np.pi):
        angle_radians += np.pi
    while (angle_radians >= np.pi):
        angle_radians -= np.pi
    if (np.pi/2 < angle_radians < np.pi):
        angle_radians -= np.pi
    if (-np.pi/2 > angle_radians >= -np.pi):
        angle_radians += np.pi
    if abs(angle_radians) <= (math.pi / 4):
        return angle_degrees
    if math.pi / 4 < angle_radians <= math.pi / 2:
        angle_radians = (math.pi / 2) - angle_radians
    if -math.pi / 4 > angle_radians >= -math.pi / 2:
        angle_radians = (- math.pi / 2) - angle_radians
    return angle_radians

print(np.degrees(normalize_angle(np.radians(-73))))
print(np.degrees(normalize_angle(np.radians(105))))