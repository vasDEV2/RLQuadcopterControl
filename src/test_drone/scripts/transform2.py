import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_frd_ned_to_rfu_enu(q_frd_ned_wxyz):
    """
    Converts a quaternion (FRD->NED) to a Rotation Matrix (RFU->ENU) using Scipy.
    
    Args:
        q_frd_ned_wxyz: list or array of shape (4,) in [w, x, y, z] format.
        
    Returns:
        numpy.ndarray: 3x3 Rotation Matrix representing RFU->ENU.
    """
    
    # 1. Convert Input [w, x, y, z] -> Scipy [x, y, z, w]
    # Scipy expects scalar-last format.
    # w, x, y, z = q_frd_ned_wxyz
    r_frd_ned = R.from_quat(q_frd_ned_wxyz)
    
    # 2. Get the Rotation Matrix from the Scipy object
    # This represents R_old
    matrix_frd_ned = r_frd_ned.as_matrix()
    
    # 3. Define the Frame Transformation Matrices
    # World: NED -> ENU
    # (North->Y, East->X, Down->-Z)
    T_ned_to_enu = np.array([
        [0, 1,  0],
        [1, 0,  0],
        [0, 0, -1]
    ])
    
    # Body: RFU -> FRD
    # (Right->Y, Front->X, Up->-Z)
    T_rfu_to_frd = np.array([
        [0, 1,  0],
        [1, 0,  0],
        [0, 0, -1]
    ])
    
    # 4. Apply the Similarity Transformation
    # R_new = T_world * R_old * T_body
    matrix_rfu_enu = T_ned_to_enu @ matrix_frd_ned @ T_rfu_to_frd

    print(f"AFTER : {R.from_matrix(matrix_rfu_enu).as_euler('ZYX', degrees=True)}")
    
    return matrix_rfu_enu

# --- Verification ---

# 1. Identity Quaternion in FRD->NED [w=1, x=0, y=0, z=0]
# Meaning: Body Front=North, Body Right=East, Body Down=Down
q_identity = [0, 0, 0, 1]

q_drone = [-0.02811205, 0.0252843, 0.7427011, 0.66855484]

print("BEFORE: ", R.from_quat(q_drone).as_euler("ZYX", degrees=True))

result = transform_frd_ned_to_rfu_enu(q_drone)

print("Input matrix (FRD->NED):", R.from_quat(q_drone).as_matrix())
print("-" * 30)
print("Resulting Matrix (RFU->ENU):")
print(result.round(1)) # Rounding for clean output
print("-" * 30)

# 2. Check if it makes sense:
# If our body was aligned FRD to NED, then in the new frame:
# RFU X (Right) is East (World X) -> should be [1, 0, 0]
# RFU Y (Front) is North (World Y) -> should be [0, 1, 0]
# RFU Z (Up)    is Up (World Z)    -> should be [0, 0, 1]
# Result should be Identity matrix.
if np.allclose(result, np.eye(3)):
    print("SUCCESS: Identity Input correctly mapped to Identity Output.")
else:
    print("CHECK: Result does not match expected Identity mapping.")