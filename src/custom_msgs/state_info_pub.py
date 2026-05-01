#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from custom_msgs.msg import StateInfo
from math import atan2, asin
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_info_publisher')

        qos_profile_topic = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publisher
        self.state_pub = self.create_publisher(StateInfo, 'drone/state_info', 10)

        # Subscribers
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos_profile_topic)
        self.create_subscription(Odometry, '/mavros/local_position/odom', self.odom_cb, qos_profile_topic)
        self.create_subscription(Imu, '/mavros/imu/data', self.imu_cb, qos_profile_topic)
        self.create_subscription(Vector3Stamped, '/deltau', self.tau_cb, 10)
        # self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        # Data holders
        self.latest_pose = None
        self.latest_vel = None
        self.latest_imu = None
        self.tau = None

        # Joint state data
        self.prev_joint_vel = [0.0, 0.0]
        self.prev_time = None
        self.joint_pos = [0.0, 0.0]
        self.joint_vel = [0.0, 0.0]
        self.joint_effort = [0.0, 0.0]
        self.joint_acc = [0.0, 0.0]

        # Timer
        self.timer = self.create_timer(0.03, self.publish_state)

        self.get_logger().info('State Info Publisher node started')

    # --- Callbacks ---
    def pose_cb(self, msg: PoseStamped):
        self.latest_pose = msg

    def odom_cb(self, msg: Odometry):
        self.latest_vel = msg

    def imu_cb(self, msg: Imu):
        self.latest_imu = msg

    def tau_cb(self, msg: Vector3Stamped):
        self.tau = msg

    # def joint_cb(self, msg: JointState):
    #     try:
    #         # Find indices for joint_1 and joint_2
    #         idx1 = msg.name.index('joint_1')
    #         idx2 = msg.name.index('joint_2')
    #     except ValueError:
    #         self.get_logger().warn("joint_1 or joint_2 not found in JointState message")
    #         return

    #     self.joint_pos[0] = msg.position[idx1]
    #     self.joint_pos[1] = msg.position[idx2]

    #     self.joint_vel[0] = msg.velocity[idx1]
    #     self.joint_vel[1] = msg.velocity[idx2]

    #     self.joint_effort[0] = msg.effort[idx1]
    #     self.joint_effort[1] = msg.effort[idx2]

    #     # Compute acceleration
    #     current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    #     if self.prev_time is not None:
    #         dt = current_time - self.prev_time
    #         if dt > 0:
    #             self.joint_acc[0] = (self.joint_vel[0] - self.prev_joint_vel[0]) / dt
    #             self.joint_acc[1] = (self.joint_vel[1] - self.prev_joint_vel[1]) / dt

    #     # Store for next iteration
    #     self.prev_joint_vel = self.joint_vel.copy()
    #     self.prev_time = current_time

    # --- Quaternion → roll, pitch, yaw ---
    def quat_to_euler(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = (3.14159265 / 2) * (1 if sinp >= 1 else (-1 if sinp <= -1 else asin(sinp)))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # --- Main publisher ---
    def publish_state(self):
        if self.latest_pose is None or self.latest_vel is None or self.latest_imu is None or self.tau is None:
            return

        msg = StateInfo()

        # Time
        msg.sec = self.latest_pose.header.stamp.sec
        msg.nsecs = self.latest_pose.header.stamp.nanosec

        # Position
        msg.x_curr = self.latest_pose.pose.position.x
        msg.y_curr = self.latest_pose.pose.position.y
        msg.z_curr = self.latest_pose.pose.position.z

        # Velocity
        msg.x_dot_curr = self.latest_vel.twist.twist.linear.x
        msg.y_dot_curr = self.latest_vel.twist.twist.linear.y
        msg.z_dot_curr = self.latest_vel.twist.twist.linear.z

        # Acceleration (IMU)
        msg.x_ddot_curr = self.latest_imu.linear_acceleration.x
        msg.y_ddot_curr = self.latest_imu.linear_acceleration.y
        msg.z_ddot_curr = self.latest_imu.linear_acceleration.z

        # Orientation
        roll, pitch, yaw = self.quat_to_euler(self.latest_pose.pose.orientation)
        msg.roll_curr = roll
        msg.pitch_curr = pitch
        msg.yaw_curr = yaw

        msg.roll_dot_curr = 0.0
        msg.pitch_dot_curr = 0.0
        msg.yaw_dot_curr = 0.0

        # Torques and thrust
        msg.tau_x = self.tau.vector.x
        msg.tau_y = self.tau.vector.y
        msg.tau_z = self.tau.vector.z
        msg.collective_thrust = 0.0

        # --- Joint states ---
        # msg.j1_pos = self.joint_pos[0]
        # msg.j2_pos = self.joint_pos[1]

        # msg.j1_vel = self.joint_vel[0]
        # msg.j2_vel = self.joint_vel[1]

        # msg.j1_acc = self.joint_acc[0]
        # msg.j2_acc = self.joint_acc[1]

        # msg.j1_torque = self.joint_effort[0]
        # msg.j2_torque = self.joint_effort[1]

        # Publish
        self.state_pub.publish(msg)


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
