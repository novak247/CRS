import numpy as np
from ctu_crs import CRS93
from pypylon import pylon
from basler_camera import BaslerCamera
import cv2
import math

def move_robot_to_xyz(robot, q0, xyz_offset):
    current_pose = robot.fk(robot.get_q())
    current_pose[:3, 3] += xyz_offset
    ik_sols = robot.ik(current_pose)
    if len(ik_sols) == 0:
        print("No IK solution found for the given offset.")
        return False
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(closest_solution)
    robot.wait_for_motion_stop()
    return True

def main():
    camera: BaslerCamera = BaslerCamera()
    camera.connect_by_name("camera-crs93")
    camera.open()
    camera.set_parameters()
    camera.start()
    robot = CRS93()
    robot.initialize()
    q0 = robot.q_home
    robot.move_to_q(robot.get_q() + [0, 0, -math.pi/4, 0, -math.pi/4, 0])
    robot.wait_for_motion_stop()

    x = np.arange(-0.2, 0.05, 0.05)
    y = np.arange(-0.2, 0.2, 0.1)
    z = np.arange(-0.1, 0.1, 0.1)

    # while True:
    #     # Input target XYZ offsets
    #     try:
    #         print(robot.fk_flange_pos(robot.get_q()))
    #         x = float(input("Enter X offset (m): "))
    #         y = float(input("Enter Y offset (m): "))
    #         z = float(input("Enter Z offset (m): "))
    #     except ValueError:
    #         print("Invalid input. Please enter numerical values.")
    #         continue

    #     xyz_offset = np.array([x, y, z])

    #     # Move robot and take image
    #     try:
    #         if move_robot_to_xyz(robot, q0, xyz_offset):
    #             print(f"Robot moved to the new position with offset: {xyz_offset}")
    #             img = camera.grab_image()
    #             if (img is not None) and (img.size > 0):
    #                 x, y, z = robot.fk_flange_pos(robot.get_q())
    #                 cv2.imwrite(f"images/X{x}Y{y}Z{z}.png", img)
    #         else:
    #             print("Robot movement failed.")
    #     except AssertionError:
    #         print("Asi spatny limity bracho")
    #         continue

    #     # Ask if user wants to continue
    #     cont = input("Do you want to move the robot again? (yes/no): ").strip().lower()
    #     if cont != 'yes':
    #         break

    robot.close()

if __name__ == "__main__":
    main()
