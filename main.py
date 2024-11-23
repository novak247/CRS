import numpy as np
from ctu_crs import CRS97, CRS93
from pypylon import pylon
from basler_camera import BaslerCamera
import cv2
import math
import csv
import time
from detect_board import HoleTransformer

def move_robot_to_xyz(robot, qnow, xyz_offset):
  current_pose = robot.fk(qnow)
  current_pose[:3, 3] += xyz_offset
  ik_sols = robot.ik(current_pose)
  if len(ik_sols) == 0:
    print("No IK solution found for the given offset.")
    return False
  closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - qnow))
  robot.move_to_q(closest_solution)
  robot.wait_for_motion_stop()
  return True

def rotate_joint(robot, q_current, joint_5_angle, joint_6_angle):
  """Rotates joint 5 and joint 6 to specified angles."""
  q_current[4] = joint_5_angle  # Set joint 5
  q_current[5] = joint_6_angle  # Set joint 6
  robot.move_to_q(q_current)
  robot.wait_for_motion_stop()

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
  robot.move_to_q(robot.get_q() + [math.pi/2, 0, 0, 0, 0, 0])
  robot.wait_for_motion_stop()
  img = camera.grab_image()
  cv2.imwrite(f"imgs/holeImage.png", img)
  robot.move_to_q(robot.get_q() - [math.pi/2, 0, 0, 0, 0, 0])
  robot.wait_for_motion_stop()

  transformer = HoleTransformer("T_base_to_camera.npy", "calibration_data.npz")
  transformed_holes, img, T_camera_to_board = transformer.detect_and_transform_holes(
      image_path="imgs/holeImage.png"
  )

  for pos in transformed_holes:
    print(pos)

  q_now = robot.get_q()
  offset = transformed_holes[0]-robot.fk(q_now)[:3,3]+np.array([0,0,0.10])
  move_robot_to_xyz(robot, q_now, offset)
  # Input target XYZ offsets
  # try:
  #   print(robot.fk_flange_pos(robot.get_q()))
  #   x1 = float(input("Enter X offset (m): "))
  #   y1 = float(input("Enter Y offset (m): "))
  #   z1 = float(input("Enter Z offset (m): "))
  # except ValueError:
  #   print("Invalid input. Please enter numerical values.")
  # q_now = robot.get_q()
  # move_robot_to_xyz(robot, q_now, np.array([x1, y1, z1]))
  # # Define the bounds for the total offsets from the initial position
  # x_bounds = np.arange(-0.2, 0.05, 0.05)
  # y_bounds = np.arange(-0.2, 0.2, 0.1)
  # z_bounds = np.arange(0.0, 0.15, 0.05)
  # joint_6_angles = np.arange(-np.pi/2, np.pi/2 + np.pi/4, np.pi/4)
  # joint_5_angles = np.arange(-np.pi/8, np.pi/8 + np.pi/16, np.pi/16)

 

    # Main iteration loop
#   with open("transformations.csv", mode="w", newline="") as file:
#     writer = csv.writer(file)
#     # Write the header
#     writer.writerow(["x", "y", "z", "joint_5", "joint_6"] + [f"T_bg_{i}" for i in range(16)])
#     q_now = robot.get_q()
#     for i in x_bounds:
#       for j in y_bounds:
#         for k in z_bounds:
          
#           # Calculate the desired relative offset
#           offset = np.array([i, j, k])
#           try:
#             # Move the robot to the new XYZ position
#             if move_robot_to_xyz(robot, q_now, offset):
#               print(f"Robot moved to the new position with total offset: {offset}")
#               q_current = robot.get_q()
#               # Iterate through joint 5 and joint 6 angles
#               for q5 in joint_5_angles:
#                 for q6 in joint_6_angles:
#                   if k == 0 and q5 == -np.pi/8:
#                     continue
#                   robot.move_to_q(q_current + [0, 0, 0, 0, q5, q6])
#                   robot.wait_for_motion_stop()
#                   time.sleep(0.1)
#                   img = camera.grab_image()
#                   if (img is not None) and (img.size > 0):
#                     x2, y2, z2 = robot.fk_flange_pos(robot.get_q()) 
#                     cv2.imwrite(
#                         f"images/X{x2}Y{y2}Z{z2}_J5{q5}_J6{q6}.png", img
#                     )
#                     T_bg = robot.fk(robot.get_q())
#                     flattened_T_bg = T_bg.flatten()
#                     writer.writerow([*offset, q5, q6, *flattened_T_bg])
#               robot.move_to_q(q_current)
#             else:
#                 print("Robot movement failed.")
#           except AssertionError:
#             print("Asi spatny limity bracho")
#             continue

#   robot.close()

if __name__ == "__main__":
    main()





# "for joint_5 in joint_5_angles:
#                 try:
#                   rotate_joint(robot, q_current, joint_5, 0)
#                   print(f"Rotated to joint_5: {joint_5}, joint_6: {0}")
                  
#                   # Capture and save the image
#                   img = camera.grab_image()
#                   if (img is not None) and (img.size > 0):
#                     x2, y2, z2 = robot.fk_flange_pos(robot.get_q()) 
#                     cv2.imwrite(
#                         f"images/X{x2}Y{y2}Z{z2}_J5{joint_5}_J6{0}.png", img
#                     )
#                     T_bg = robot.fk(robot.get_q())
#                     flattened_T_bg = T_bg.flatten()
#                     writer.writerow([*offset, joint_5, 0, *flattened_T_bg])  
#                 except AssertionError:
#                   print(f"Failed to rotate to joint_5: {joint_5}, joint_6: {0}")
                  
#               try:
#                 rotate_joint(robot, q_current, 0, 0)
#               except:
#                 print("Failed to rotate to joint_5: 0, joint_6: 0")
#               for joint_6 in joint_6_angles:
#                 try:
#                   rotate_joint(robot, q_current, 0, joint_6)
#                   print(f"Rotated to joint_5: {0}, joint_6: {joint_6}")
                  
#                   # Capture and save the image
#                   img = camera.grab_image()
#                   if (img is not None) and (img.size > 0):
#                     x2, y2, z2 = robot.fk_flange_pos(robot.get_q()) 
#                     cv2.imwrite(
#                         f"images/X{x2}Y{y2}Z{z2}_J5{0}_J6{joint_6}.png", img
#                     )
#                     T_bg = robot.fk(robot.get_q())
#                     flattened_T_bg = T_bg.flatten()
#                     writer.writerow([*offset, 0, joint_6, *flattened_T_bg])  
#                 except AssertionError:
#                   print(f"Failed to rotate to joint_5: {0}, joint_6: {joint_6}")
#               try:
#                 rotate_joint(robot, q_current, 0, 0)
#               except:
#                 print("Failed to rotate to joint_5: 0, joint_6: 0")"