# StateInfo.msg
# Custom message for current state and torque values

# Timestamp
int64 sec
int64 nsecs

# Current position (m)
float64 x_curr
float64 y_curr
float64 z_curr

# Current velocity (m/s)
float64 x_dot_curr
float64 y_dot_curr
float64 z_dot_curr

# Current acceleration (m/s^2)
float64 x_ddot_curr
float64 y_ddot_curr
float64 z_ddot_curr

# Current attitude (radians)
float64 roll_curr
float64 pitch_curr
float64 yaw_curr

float64 roll_dot_curr
float64 pitch_dot_curr
float64 yaw_dot_curr

float64 ex
float64 ey
float64 ez

float64 ex_dot
float64 ey_dot
float64 ez_dot

float64 ex_ddot
float64 ey_ddot
float64 ez_ddot

float64 roll_err
float64 pitch_err
float64 yaw_err

# Current torques (Nm)
float64 tau_x
float64 tau_y
float64 tau_z

float64 collective_thrust

#float64 j1_torque
#float64 j2_torque

#float64 j1_pos
#float64 j2_pos

#float64 j1_vel
#float64 j2_vel

#float64 j1_acc
#float64 j2_acc



