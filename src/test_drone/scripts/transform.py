import numpy as np
from scipy.spatial.transform import Rotation as R

def ned_to_frd_vector(q_frd_to_ned, v_ned):
    """
    Convert a vector from NED frame to FRD frame using a quaternion that defines
    the rotation from FRD → NED.

    q_frd_to_ned : array_like of shape (4,)  quaternion [w, x, y, z]
    v_ned        : array_like of shape (3,)  vector in NED frame
    """
    # Create rotation object (SciPy uses [x, y, z, w])
    r_frd_to_ned = R.from_quat([q_frd_to_ned[1], 
                                q_frd_to_ned[2], 
                                q_frd_to_ned[3], 
                                q_frd_to_ned[0]])

    # Inverse gives NED → FRD rotation
    r_ned_to_frd = r_frd_to_ned.inv()

    # Rotate vector
    return r_ned_to_frd.apply(v_ned)


# --------------------------
# Example usage
# --------------------------
q_frd_to_ned = np.array([9.9999917e-01,  9.1201298e-05, -6.1308907e-05,  1.2989950e-03])  # quaternion [w, x, y, z]
v_ned = np.array([0.0, 0.0, -9.8])

v_frd = ned_to_frd_vector(q_frd_to_ned, v_ned)
print("Vector in FRD:", v_frd)
