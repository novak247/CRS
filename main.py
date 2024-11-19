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

    # Input target XYZ offsets
    try:
        print(robot.fk_flange_pos(robot.get_q()))
        x1 = float(input("Enter X offset (m): "))
        y1 = float(input("Enter Y offset (m): "))
        z1 = float(input("Enter Z offset (m): "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")

    move_robot_to_xyz(robot, q0, np.array([x1, y1, z1]))
    # Define the bounds for the total offsets from the initial position
    x_bounds = np.arange(-0.2, 0.05, 0.05)
    y_bounds = np.arange(-0.2, 0.2, 0.1)
    z_bounds = np.arange(-0.1, 0.1, 0.1)

    # Initialize the cumulative offset
    cumulative_offset = np.array([0.0, 0.0, 0.0])

    # Main iteration loop
    for i in x_bounds:
        for j in y_bounds:
            for k in z_bounds:
                # Calculate the desired relative offset
                relative_offset = np.array([i, j, k]) - cumulative_offset

                # Check if the total offset exceeds the bounds
                proposed_offset = cumulative_offset + relative_offset
                if (np.abs(proposed_offset) > np.array([0.2, 0.2, 0.1])).any():
                    print(f"Skipping offset {[i, j, k]} as it exceeds bounds.")
                    continue

                try:
                    # Move the robot with the relative offset
                    if move_robot_to_xyz(robot, q0, relative_offset):
                        # Update the cumulative offset after a successful move
                        cumulative_offset += relative_offset
                        print(f"Robot moved to the new position with total offset: {cumulative_offset}")
                        img = camera.grab_image()
                        if (img is not None) and (img.size > 0):
                            x2, y2, z2 = robot.fk_flange_pos(robot.get_q())
                            cv2.imwrite(f"images/X{x2}Y{y2}Z{z2}.png", img)
                    else:
                        print("Robot movement failed.")
                except AssertionError:
                    print("Asi spatny limity bracho")
                    continue

    robot.close()

if __name__ == "__main__":
    main()
