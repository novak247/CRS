import numpy as np
from ctu_crs import CRS93

robot = CRS93()
q = robot.get_q()
print(robot.fk_flange_pos(q))