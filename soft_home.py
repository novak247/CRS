import numpy as np
from ctu_crs import CRS93

robot = CRS93()
robot.initialize()

#[ 2.33481871e-01 -1.83376233e-05  8.43486659e-01] XYZ of home position

robot.soft_home()