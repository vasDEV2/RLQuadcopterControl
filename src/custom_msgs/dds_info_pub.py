#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from math import atan2, asin
import math

from geometry_msgs.msg import Vector3Stamped
from trajectory_msgs.msg import MultiDOFJointTrajectory
from px4_msgs.msg import VehicleOdometry, VehicleAcceleration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from custom_msgs.msg import StateInfo


class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_info_publisher')

        sim_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # ---------------- Publisher ----------------
        self.state_pub = self.create_publisher(StateInfo, 'drone/state_info', 10)

        # ---------------- Subscribers ----------------
        self.create_subscription(
            VehicleAcceleration,
            '/fmu/out/vehicle_acceleration',
            self.accel_callback,
            qos_profile=sim_qos_profile
        )

        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_sub_callback,
            qos_profile=sim_qos_profile
        )

        self.create_subscription(
            Vector3Stamped,
            '/deltau',
            self.tau_cb,
            10
        )

        self.create_subscription(
            MultiDOFJointTrajectory,
            'command/trajectory',
            self.trajectory_cb,
            10
        )

        # ---------------- Current State ----------------
        self.curr_position = [0.0, 0.0, 0.0]
        self.curr_linear_vel = [0.0, 0.0, 0.0]
        self.curr_linear_acc = [0.0, 0.0, 0.0]
        self.curr_orientation = [1.0, 0.0, 0.0, 0.0]  # x y z w
        self.curr_rpy = [0.0, 0.0, 0.0]

        # ---------------- Desired State ----------------
        self.des_position = [0.0, 0.0, 0.0]
        self.des_linear_vel = [0.0, 0.0, 0.0]
        self.des_linear_acc = [0.0, 0.0, 0.0]
        self.des_rpy = [0.0, 0.0, 0.0]

        self.traj_received = False
        self.tau = None
        self.time = 0

        # ---------------- Timer ----------------
        self.timer = self.create_timer(0.03, self.publish_state)

        self.get_logger().info('State Info Publisher with trajectory tracking started')

    # =========================================================
    # Callbacks
    # =========================================================

    def accel_callback(self, msg: VehicleAcceleration):
        self.curr_linear_acc[0] = msg.xyz[0]
        self.curr_linear_acc[1] = -msg.xyz[1]
        self.curr_linear_acc[2] = -msg.xyz[2]

    def odom_sub_callback(self, msg: VehicleOdometry):
        self.time = msg.timestamp

        self.curr_position[0] = msg.position[0]
        self.curr_position[1] = -msg.position[1]
        self.curr_position[2] = -msg.position[2]

        self.curr_linear_vel[0] = msg.velocity[0]
        self.curr_linear_vel[1] = -msg.velocity[1]
        self.curr_linear_vel[2] = -msg.velocity[2]

        # PX4 quaternion → ENU
        self.curr_orientation = [
            msg.q[1],
            -msg.q[2],
            -msg.q[3],
            msg.q[0]
        ]

        self.curr_rpy = self.quat_to_euler(self.curr_orientation)

    def tau_cb(self, msg: Vector3Stamped):
        self.tau = msg

    def trajectory_cb(self, msg: MultiDOFJointTrajectory):
        if not msg.points:
            return

        pt = msg.points[0]
        if not pt.transforms:
            return

        # Desired position
        self.des_position[0] = pt.transforms[0].translation.x
        self.des_position[1] = pt.transforms[0].translation.y
        self.des_position[2] = pt.transforms[0].translation.z

        # Desired velocity
        if pt.velocities:
            self.des_linear_vel[0] = pt.velocities[0].linear.x
            self.des_linear_vel[1] = pt.velocities[0].linear.y
            self.des_linear_vel[2] = pt.velocities[0].linear.z

        # Desired acceleration
        if pt.accelerations:
            self.des_linear_acc[0] = pt.accelerations[0].linear.x
            self.des_linear_acc[1] = pt.accelerations[0].linear.y
            self.des_linear_acc[2] = pt.accelerations[0].linear.z

        # Desired orientation
        q = pt.transforms[0].rotation
        self.des_rpy = self.quat_to_euler([q.x, q.y, q.z, q.w])

        self.traj_received = True

    # =========================================================
    # Utilities
    # =========================================================

    def quat_to_euler(self, q):
        x, y, z, w = q

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = atan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

    def wrap_angle(self, a):
        return atan2(math.sin(a), math.cos(a))

    # =========================================================
    # Main Publisher
    # =========================================================

    def publish_state(self):
        if not self.traj_received or self.tau is None:
            return

        msg = StateInfo()

        # ---------------- Time ----------------
        msg.sec = 0
        msg.nsecs = self.time

        # ---------------- Current State ----------------
        msg.x_curr = float(self.curr_position[0])
        msg.y_curr = float(self.curr_position[1])
        msg.z_curr = float(self.curr_position[2])

        msg.x_dot_curr = float(self.curr_linear_vel[0])
        msg.y_dot_curr = float(self.curr_linear_vel[1])
        msg.z_dot_curr = float(self.curr_linear_vel[2])

        msg.x_ddot_curr = float(self.curr_linear_acc[0])
        msg.y_ddot_curr = float(self.curr_linear_acc[1])
        msg.z_ddot_curr = float(self.curr_linear_acc[2])

        msg.roll_curr = float(self.curr_rpy[0])
        msg.pitch_curr = float(self.curr_rpy[1])
        msg.yaw_curr = float(self.curr_rpy[2])


        msg.roll_dot_curr = 0.0
        msg.pitch_dot_curr = 0.0
        msg.yaw_dot_curr = 0.0

        # ---------------- Errors ----------------
        msg.ex = float(self.des_position[0] - self.curr_position[0])
        msg.ey = float(self.des_position[1] - self.curr_position[1])
        msg.ez = float(self.des_position[2] - self.curr_position[2])

        msg.ex_dot = float(self.des_linear_vel[0] - self.curr_linear_vel[0])
        msg.ey_dot = float(self.des_linear_vel[1] - self.curr_linear_vel[1])
        msg.ez_dot = float(self.des_linear_vel[2] - self.curr_linear_vel[2])

        msg.ex_ddot = float(self.des_linear_acc[0] - self.curr_linear_acc[0])
        msg.ey_ddot = float(self.des_linear_acc[1] - self.curr_linear_acc[1])
        msg.ez_ddot = float(self.des_linear_acc[2] - self.curr_linear_acc[2])

        msg.roll_err = float(self.des_rpy[0] - self.curr_rpy[0])
        msg.pitch_err = float(self.des_rpy[1] - self.curr_rpy[1])
        msg.yaw_err = float(self.wrap_angle(self.des_rpy[2] - self.curr_rpy[2]))


        # ---------------- Control ----------------
        msg.tau_x = self.tau.vector.x
        msg.tau_y = self.tau.vector.y
        msg.tau_z = self.tau.vector.z
        msg.collective_thrust = 0.0

        self.state_pub.publish(msg)


# =========================================================
# Main
# =========================================================

def main(args=None):
    rclpy.init(args=args)
    node = StatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
