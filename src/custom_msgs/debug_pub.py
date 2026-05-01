#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from custom_msgs.msg import DebugInfo
from math import atan2, asin
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math
import numpy as np


class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_info_publisher')

        qos_profile_topic = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publisher
        self.state_pub = self.create_publisher(DebugInfo, '/debug_info', 10)

        # Subscribers
        self.create_subscription(Vector3Stamped, '/H_pred', self.pred_cb, 10)
        self.create_subscription(Vector3Stamped, '/H_actual', self.actual_cb, 10)
        self.create_subscription(Vector3Stamped, '/deltau', self.tau_cb, 10)
        self.create_subscription(Vector3Stamped, '/acc', self.acc_cb, 10)
        # self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        # Data holders
        self.h_pred = None
        self.h_des = None
        self.tau = None
        self.rmse = 0.0

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
    def pred_cb(self, msg: Vector3Stamped):
        self.h_pred = msg

    def actual_cb(self, msg: Vector3Stamped):
        self.h_des = msg

    def tau_cb(self, msg: Vector3Stamped):
        self.tau = msg

    def acc_cb(self, msg: Vector3Stamped):
        self.acc = msg

    def calculate_rmse(self, h_P, h_d):
        """Calculate RMSE between predicted and desired angular momentum"""
        # Extract coordinates into numpy arrays
        p = np.array([h_P.vector.x, h_P.vector.y, h_P.vector.z])
        d = np.array([h_d.vector.x, h_d.vector.y, h_d.vector.z])
        
        # Calculate and return RMSE
        return np.float64(np.sqrt(np.mean((p - d)**2)))
    
    

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
        if self.h_pred is None or self.h_des is None or self.tau is None:
            return
        
        msg = DebugInfo()
        
        # Time
        msg.sec = self.tau.header.stamp.sec
        msg.nsecs = self.tau.header.stamp.nanosec
        
        # Position
        msg.h_pred_x = self.h_pred.vector.x
        msg.h_pred_y = self.h_pred.vector.y
        msg.h_pred_z = self.h_pred.vector.z
        
        msg.h_des_x = self.h_des.vector.x
        msg.h_des_y = self.h_des.vector.y
        msg.h_des_z = self.h_des.vector.z
        
        # Torques and thrust
        msg.tau_x = self.tau.vector.x
        msg.tau_y = self.tau.vector.y
        msg.tau_z = self.tau.vector.z
        
        msg.acc_x = self.acc.vector.x
        msg.acc_y = self.acc.vector.y
        msg.acc_z = self.acc.vector.z
        
        # Calculate RMSE
        self.rmse = self.calculate_rmse(self.h_pred, self.h_des)
        msg.rmse = self.rmse
        
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
