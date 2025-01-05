import numpy as np
from ctu_crs import CRS97, CRS93
from pypylon import pylon
from basler_camera import BaslerCamera
import cv2
import math
import csv
import time
from detect_board import HoleTransformer
import os


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
  robot.move_to_q(robot.get_q() + [0, 0, -math.pi/4, 0, -math.pi/4, 0])
  robot.wait_for_motion_stop()
  robot.move_to_q(robot.get_q() + [0.95*math.pi, 0, 0, 0, 0, 0])
  robot.wait_for_motion_stop()
  getImages(camera, robot, 20)
  robot.move_to_q(robot.get_q() - [0.95*math.pi, 0, 0, 0, 0, 0])
  robot.wait_for_motion_stop()

def extract_axis_angle(R_base_to_board):
  rvec, _ = cv2.Rodrigues(R_base_to_board)
  angle = np.linalg.norm(rvec) 
  axis = rvec / angle if not np.isclose(angle, 0)  else np.array([0, 0, 0])
  return axis, angle * (180 / math.pi)

def decompose_rotation(R):
    """
    Decomposes a rotation matrix into a rotation around the z-axis and another rotation.
    
    Parameters:
        R (np.ndarray): A 3x3 rotation matrix.
        
    Returns:
        theta_z (float): Angle of rotation around the z-axis in radians.
        axis (np.ndarray): The axis of the second rotation (3D vector).
        angle (float): Angle of the second rotation in radians.
    """
    # Ensure the input is a valid rotation matrix
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix"
    assert np.allclose(np.dot(R.T, R), np.eye(3)), "Input must be orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0), "Determinant of the rotation matrix must be 1"

    # Extract rotation angle around the z-axis
    theta_z = np.arctan2(R[1, 0], R[0, 0])
    
    # Compute R_z(-theta_z)
    R_z_neg_theta = np.array([
        [np.cos(-theta_z), -np.sin(-theta_z), 0],
        [np.sin(-theta_z), np.cos(-theta_z),  0],
        [0,               0,                1]
    ])
    
    # Compute the remaining rotation matrix
    R_u = np.dot(R_z_neg_theta, R)

    # Compute the angle of rotation for R_u
    angle = np.arccos((np.trace(R_u) - 1) / 2)

    # Compute the rotation axis from R_u
    axis = np.array([
        R_u[2, 1] - R_u[1, 2],
        R_u[0, 2] - R_u[2, 0],
        R_u[1, 0] - R_u[0, 1]
    ])
    axis = axis / (2 * np.sin(angle))

    return theta_z, axis, angle

def normalize_angle(angle):
  angle_normalized = (angle + np.pi) % (2 * np.pi) - np.pi
  if angle_normalized > np.pi/2:
    angle_normalized = -np.pi + angle_normalized
  elif angle < -np.pi/2:
    angle_normalized = np.pi + angle_normalized
  return angle_normalized

def main():
  camera, robot = initialize_robot_and_camera()
  get_board_position_images(robot, camera) 

  transformer = HoleTransformer("T_base_to_camera_new.npy", "calibration_data.npz")

  img_path = "board_imgs"
  hole_positions = get_hole_positions(transformer, img_path)

  R1 = (transformer.T_base_to_camera @ transformer.T_camera_to_board_lower[0])[:3,:3]
  R2 = (transformer.T_base_to_camera @ transformer.T_camera_to_board_lower[1])[:3,:3]
  # print("decomposed rotation R1: ", normalize_angle(decompose_rotation(R1)[0])*180/np.pi)
  # print("decomposed rotation R2: ", normalize_angle(decompose_rotation(R2)[0])*180/np.pi)
  theta1 = normalize_angle(decompose_rotation(R1)[0])
  theta2 = normalize_angle(decompose_rotation(R2)[0])
  transformed_holes = []
  print(np.array([hole_positions[i][0] for i in range(len(hole_positions))]))
  transformed_holes.append(transformer.refine_hole_positions(np.array([hole_positions[i][0] for i in range(len(hole_positions))])))
  transformed_holes.append(transformer.refine_hole_positions(np.array([hole_positions[i][1] for i in range(len(hole_positions))])))

  pairs = make_pairs(transformed_holes)
  for pair in enumerate(pairs):

    print(f"Pair {pair[0]+1}: ", *pair[1])
  
  homePosition = robot.get_q()
  for pair in pairs:
    switch = 1
    for pos in reversed(pair):
      if switch == 1:
        offset = pos-robot.fk(homePosition)[:3,3]+np.array([-0.03,0.0,0.10])
        if pos[1] < 0:
          offset += np.array([0.0, 0.0075, 0.0])
        print("Position:", pos)
        move_robot_to_xyz(robot, homePosition, offset)
        move_robot_to_xyz(robot, homePosition, offset-np.array([0, 0, 0.05]))
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
        print("Position:", pos)
        move_robot_to_xyz(robot, homePosition, offset)
        move_robot_to_xyz(robot, homePosition, offset-np.array([0, 0, 0.04]))
        # robot.move_to_q(np.append(robot.get_q()[:5], -theta1))
        # robot.wait_for_motion_stop()
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