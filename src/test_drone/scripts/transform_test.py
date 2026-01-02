import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_frd_ned_to_flu_enu(quat_frd_ned):
    """
    Converts a quaternion from FRD->NED to FLU->ENU using Scipy.
    
    Args:
        quat_frd_ned: List or Array [x, y, z, w] representing the drone's rotation.
        
    Returns:
        quat_flu_enu: Array [x, y, z, w] for the RL policy.
    """
    
    # 1. Define the Input Rotation
    # (The raw orientation from the drone)
    r_input = R.from_quat(quat_frd_ned)
    
    # 2. Define the Body Frame Fix (FLU -> FRD)
    # Rotation of 180 degrees (pi radians) around the X-axis
    r_body_fix = R.from_euler('x', 180, degrees=True)
    
    # 3. Define the World Frame Fix (NED -> ENU)
    # Rotation of 180 degrees around the X=Y diagonal axis.
    # We construct this explicitly using the quaternion derived in the math section.
    # q = [x=0.707, y=0.707, z=0, w=0]
    c = 1 / np.sqrt(2)
    r_world_fix = R.from_quat([c, c, 0, 0])
    
    # 4. Multiply them in the correct order:
    # Scipy Rotation multiplication: C = A * B means "Apply B, then Apply A"
    # Order of operations on vector v: 
    # v_enu = R_world_fix * (R_input * (R_body_fix * v_flu))
    
    r_final = r_world_fix * r_input * r_body_fix

    print("AFTER: ", r_final.as_euler("ZYX", degrees=True))
    
    return r_final.as_quat()

# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def get_flattened_obs(q_frd_ned, fix_pitch_axis=True):
#     """
#     Converts FRD (Drone) Quaternion to the Flat Rotation Matrix expected by RL.
    
#     Args:
#         q_frd_ned: [x, y, z, w] from the drone (Scipy format)
#         fix_pitch_axis: Set True if your policy mixes up Pitch/Roll (Pitch=X).
#                         Set False if your policy uses standard FLU (Pitch=Y).
    
#     Returns:
#         rotation_matrix_flat: A numpy array of shape (9,)
#     """
    
#     # 1. Load the drone's orientation
#     r_drone = R.from_quat(q_frd_ned)
    
#     # 2. Standard Conversion: FRD -> FLU
#     # Rotate 180 deg around X-axis to fix Up/Down inversion
#     r_body_fix = R.from_euler('x', 180, degrees=True)
    
#     # 3. World Conversion: NED -> ENU
#     # Rotate 180 deg around X=Y axis to fix North/East inversion
#     c = 0.70710678
#     r_world_fix = R.from_quat([c, c, 0, 0])
    
#     # 4. (Optional) The "Pitch = Index 2" Fix
#     # If your policy thinks Pitch is X-axis, we need to rotate the body 
#     # frame -90 degrees around Z so the Lateral axis aligns with X.
#     if fix_pitch_axis:
#         # Rotates Body Frame -90 deg (Clockwise)
#         r_policy_alignment = R.from_euler('z', -90, degrees=True)
#     else:
#         r_policy_alignment = R.from_quat([0, 0, 0, 1]) # Identity

#     # 5. Combine Rotations
#     # Order: WorldFix * Drone * BodyFix * PolicyAlignment
#     r_final = r_world_fix * r_drone * r_body_fix * r_policy_alignment
    
#     # 6. Get Rotation Matrix (3x3)
#     matrix = r_final.as_matrix()

#     print(f'AFTER : {r_final.as_euler("ZYX", degrees = True)}')
    
#     # 7. Flatten to (9,)
#     # RL policies usually expect row-major flattening
#     flat_matrix = matrix.flatten() 
    
#     return flat_matrix

# # --- Example usage ---
# # Drone is Level, facing North (FRD)
# q_input = [-3.316646e-05, -0.00030439056, 0.7429813, 0.6693121] 
# print("BEFORE: ", R.from_quat(q_input).as_euler("ZYX", degrees=True))

# obs = get_flattened_obs(q_input, fix_pitch_axis=True)

# print(f"Shape: {obs.shape}")
# print(f"Data: {obs}")

# --- Example Usage ---

# Example: Drone is LEVEL facing NORTH in FRD/NED
# In Scipy [x, y, z, w], Identity is [0, 0, 0, 1]
q_drone = [0, 0, 0, 1]
# q_drone = [-3.316646e-05, -0.00030439056, 0.7429813, 0.6693121] 
q_drone = [-0.02811205, 0.0252843, 0.7427011, 0.66855484]

print("BEFORE: ", R.from_quat(q_drone).as_euler("ZYX", degrees=True))

q_rl_policy = convert_frd_ned_to_flu_enu(q_drone)

print(f"Input (FRD->NED): {q_drone}")
print(f"Output (FLU->ENU): {q_rl_policy}")

# print()

# Expected Output Analysis:
# If drone is Level/North in FRD/NED.
# In FLU/ENU, that same physical orientation means the body is upside down 
# (because Z is Up in World but Down in Body).
# The output should represent a 180 degree rotation around the vector [1, 1, 0].