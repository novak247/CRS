import numpy as np
from ctu_crs import CRS97

robot = CRS97()
# q = robot.get_q()
# print(robot.fk_flange_pos(q))

robot.initialize()
robot.move_to_q([0, 0, np.deg2rad(-90), 0, np.deg2rad(-90), 0])
robot.gripper.control_position(-1000)