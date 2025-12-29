import numpy as np
from scipy.spatial.transform import Rotation as R
import math


ang  = [0, math.radians(20), math.radians(30)]

print(f"ang {ang}")

r = R.from_euler("zyx", ang)

mat1 = r.as_matrix()

print(f"mat1: {mat1}")

R_map = np.array([
        [0,  1,  0],
        [1,  0,  0],
        [0,  0, -1]
    ])

rotated_m = R_map @ mat1 @ R_map.T

print(f"rotated: {rotated_m}")

rr = R.from_matrix(rotated_m)

# angs = rr.as_euler("zxy")

# print(f"rotated angs: {angs}")

# rrr = R.from_euler("zxy", ang)

r_m = rrr.as_matrix()

print(f"NEW {r_m}")