from scipy.spatial.transform import Rotation as R
import numpy as np


ang = np.array([0.022, 0.043, 0.223])

mat = R.from_euler("ZYX", ang)

print(mat.as_matrix())

