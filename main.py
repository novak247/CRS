import numpy as np
from ctu_crs import CRS97, CRS93
from basler_camera import BaslerCamera
import cv2
import csv
import time
from detect_board import HoleTransformer
import os
import sys
from scipy.spatial.transform import Rotation as R


def move_robot_to_xyz(robot, qnow, xyz_offset, flange_rot = False, theta = 0):
  current_pose = robot.fk(qnow)
  current_pose[:3, 3] += xyz_offset
  ik_sols = robot.ik(current_pose)
  closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - qnow))
  if flange_rot:
    closest_solution = np.append(closest_solution[:5], theta + closest_solution[0])
  robot.move_to_q(closest_solution)
  robot.wait_for_motion_stop()
  return True


def rotate_joint(robot, q_current, joint_5_angle, joint_6_angle):
  """Rotates joint 5 and joint 6 to specified angles."""
  q_current[4] = joint_5_angle  # Set joint 5
  q_current[5] = joint_6_angle  # Set joint 6
  robot.move_to_q(q_current)
  robot.wait_for_motion_stop()

def getImages(camera, robot, n):
  for i in range(20):
    img = camera.grab_image()
    cv2.imwrite(f"board_imgs/hImage{i}.png", img)
    robot.wait_for_motion_stop()
    time.sleep(0.25)

def make_pairs(holes):
  parky = []
  for i in range(len(holes[0])):
    parky.append([holes[0][i], holes[1][i]])
  return parky

def get_hole_positions(transformer, img_dir):
  hole_positions = []
  files = os.listdir(img_dir)
  imgs = [os.path.join(img_dir, f) for f in files if f.lower().endswith(".png")]
  for img in imgs:
    transformed_holes, _, _ = transformer.detect_and_transform_holes(image_path=img)
    hole_positions.append(transformed_holes)
  return hole_positions

# def get_transformations(transformer, img_dir):
#   tfs = []
#   files = os.listdir(img_dir)
#   imgs = [os.path.join(img_dir, f) for f in files if f.lower().endswith(".png")]
#   for img in imgs:
#     _, _, tf = transformer.detect_and_transform_holes(image_path=img)
#     tfs.append(transformed_holes)
#   return tfs

def initialize_robot_and_camera():
  camera: BaslerCamera = BaslerCamera()
  camera.connect_by_name("camera-crs97")
  camera.open()
  camera.set_parameters()
  robot = CRS97()
  robot.initialize()
  robot.gripper.control_position(1000)
  return camera, robot

def get_board_position_images(robot, camera):
  robot.move_to_q(robot.get_q() + [0, 0, -np.pi/4, 0, -np.pi/4, 0])
  robot.wait_for_motion_stop()
  robot.move_to_q(robot.get_q() + [0.95*np.pi, 0, 0, 0, 0, 0])
  robot.wait_for_motion_stop()
  getImages(camera, robot, 20)
  robot.move_to_q(robot.get_q() - [0.95*np.pi, 0, 0, 0, 0, 0])
  robot.wait_for_motion_stop()

def extract_axis_angle(R_base_to_board):
  rvec, _ = cv2.Rodrigues(R_base_to_board)
  angle = np.linalg.norm(rvec) 
  axis = rvec / angle if not np.isclose(angle, 0)  else np.array([0, 0, 0])
  return axis, angle * (180 / np.pi)

def decompose_rotation(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz')
    return euler_angles[2]

# def normalize_angle(angle_radians):
#     print("---------", np.degrees(angle_radians))

#     while (angle_radians < -np.pi):
#         angle_radians += np.pi
#     while (angle_radians >= np.pi):
#         angle_radians -= np.pi
#     if (np.pi/2 < angle_radians < np.pi):
#         angle_radians = np.pi - angle_radians
#     if (-np.pi/2 > angle_radians >= -np.pi):
#         angle_radians = -np.pi - angle_radians
#     if abs(angle_radians) <= (np.pi / 4):
#         return angle_radians
#     if np.pi / 4 < angle_radians <= np.pi / 2:
#         angle_radians = (np.pi / 2) - angle_radians
#     if -np.pi / 4 > angle_radians >= -np.pi / 2:
#         angle_radians = (- np.pi / 2) - angle_radians
#     return angle_radians

def normalize_angle(angle_radians):
  if angle_radians > np.pi / 2:
    angle_radians -= np.pi
  if angle_radians < - np.pi / 2:
    angle_radians += np.pi
  if abs(angle_radians - np.pi/2) < np.radians(2) or abs(angle_radians + np.pi/2) < np.radians(2):
    angle_radians = 0
  return angle_radians

def main():
  ordering = sys.argv[1]
  camera, robot = initialize_robot_and_camera()
  get_board_position_images(robot, camera) 

  transformer = HoleTransformer("T_base_to_camera_new.npy", "calibration_data.npz")

  img_path = "board_imgs"
  hole_positions = get_hole_positions(transformer, img_path)
  # tfs = get_transformations(transformer, img_path)
  R1 = (transformer.T_base_to_camera @ transformer.T_camera_to_board_lower[0])[:3,:3]
  R2 = (transformer.T_base_to_camera @ transformer.T_camera_to_board_lower[1])[:3,:3]
  # print("decomposed rotation R1: ", normalize_angle(decompose_rotation(R1)[0])*180/np.pi)
  # print("decomposed rotation R2: ", normalize_angle(decompose_rotation(R2)[0])*180/np.pi)
  theta1 = normalize_angle(decompose_rotation(R1))
  theta2 = normalize_angle(decompose_rotation(R2))
  # theta1 = decompose_rotation(R1)[0]
  # theta2 = decompose_rotation(R2)[0]
  transformed_holes = []
  print("THETAS: ", np.degrees(theta1), np.degrees(theta2))
  print(np.array([hole_positions[i][0] for i in range(len(hole_positions))]))
  transformed_holes.append(transformer.refine_hole_positions(np.array([hole_positions[i][0] for i in range(len(hole_positions))])))
  transformed_holes.append(transformer.refine_hole_positions(np.array([hole_positions[i][1] for i in range(len(hole_positions))])))

  pairs = make_pairs(transformed_holes)
  for pair in enumerate(pairs):
    print(f"Pair {pair[0]+1}: ", *pair[1])
  board_rot = transformer.board_rot
  reversed_rot = False
    
  homePosition = robot.get_q()
  for pair in pairs:
    switch = 1
    if ordering == "lr" and pair[0][1] > pair[1][1]:
      pair.reverse()
      print("pair after reversed lr", pair)
      if not reversed_rot:
        board_rot.reverse()
        reversed_rot = True
        theta1, theta2 = theta2, theta1
    elif ordering == "rl" and pair[0][1] < pair[1][1]:
      pair.reverse()
      print("pair after reversed rl", pair)
      if not reversed_rot:
        board_rot.reverse()
        reversed_rot = True
        theta1, theta2 = theta2, theta1
    print(board_rot)
    for pos in pair:
      print(pos)
      if switch == 1:
        offset = pos-robot.fk(homePosition)[:3,3]+np.array([-0.03,0.0,0.10])
        if pos[1] < 0:
          offset += np.array([0.0, 0.0075, 0.0])
        print("Position:", pos, " Angle: ", -theta1)
        move_robot_to_xyz(robot, homePosition, offset)
        move_robot_to_xyz(robot, homePosition, offset-np.array([0, 0, 0.05]))
        robot.move_to_q(robot.get_q() + np.array([0,0,0,0,0,-theta1]) )
        robot.wait_for_motion_stop()
        time.sleep(1)
        robot.gripper.control_position(-1000 * switch)
        time.sleep(2)
        switch *= -1
        move_robot_to_xyz(robot, homePosition, offset)
        robot.wait_for_motion_stop()
        move_robot_to_xyz(robot, homePosition, np.array([0, 0, 0]))
      else:
        offset = pos-robot.fk(homePosition)[:3,3]+np.array([-0.03,0,0.10])
        if pos[1] < 0:
          offset += np.array([0.0, 0.0075, 0.0])
        print("Position:", pos, " Angle: ", -theta2)
        move_robot_to_xyz(robot, homePosition, offset)
        move_robot_to_xyz(robot, homePosition, offset-np.array([0, 0, 0.03]))
        robot.move_to_q(robot.get_q() + np.array([0,0,0,0,0,-theta2]))
        robot.wait_for_motion_stop()
        time.sleep(1)
        robot.gripper.control_position(-1000 * switch)
        time.sleep(2)
        switch *= -1
        move_robot_to_xyz(robot, homePosition, offset)
        robot.wait_for_motion_stop()
        move_robot_to_xyz(robot, homePosition, np.array([0, 0, 0]))
        
  
  robot.soft_home()

if __name__ == "__main__":
    main()