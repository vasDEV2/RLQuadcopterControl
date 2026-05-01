#!/usr/bin/env python3
# ROS2 port of your rospy controller script
# Save as setpoint_node_ros2.py in a ROS2 python package (executable)
import sys
import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# messages / services
from std_msgs.msg import *
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Accel, Vector3, Vector3Stamped

from mavros_msgs.msg import *
from mavros_msgs.msg import Thrust
from mavros_msgs.srv import CommandTOL, CommandBool, SetMode
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt


from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# transformations
# In ROS2, depending on packages installed you may import from tf_transformations
try:
    import tf_transformations as tf_trans
except Exception:
    # fallback to tf.transformations if available
    try:
        from tf import transformations as tf_trans
    except Exception:
        tf_trans = None
        # If not available, you'll need to install python package `transformations` or `tf_transformations`

# cross-platform getch
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import tty, termios
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class FCUModesROS2:
    """Service wrappers for arming, takeoff, set_mode"""
    def __init__(self, node: Node):
        self.node = node
        self._takeoff_client = node.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self._arming_client = node.create_client(CommandBool, '/mavros/cmd/arming')
        self._set_mode_client = node.create_client(SetMode, '/mavros/set_mode')

    def _wait_client(self, client, timeout_sec=5.0):
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().warn(f"Service {client.srv_name} not available")
            return False
        return True

    def set_takeoff(self, altitude=3.0):
        if not self._wait_client(self._takeoff_client):
            return False
        req = CommandTOL.Request()
        # Common fields; if your version differs, inspect service definition and adjust
        try:
            req.altitude = float(altitude)
        except Exception:
            pass
        fut = self._takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        return True

    def set_arm(self, arm=True):
        if not self._wait_client(self._arming_client):
            return False
        req = CommandBool.Request()
        req.value = bool(arm)
        fut = self._arming_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        return True

    def set_mode(self, custom_mode: str):
        if not self._wait_client(self._set_mode_client):
            return False
        req = SetMode.Request()
        # Many SetMode services accept "custom_mode" or "base_mode". Adjust if your srv differs.
        try:
            req.custom_mode = str(custom_mode)
        except Exception:
            # some flavors have 'mode' field
            try:
                req.mode = str(custom_mode)
            except Exception:
                pass
        fut = self._set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        return True


class Controller(Node):
    def __init__(self):
        super().__init__('setpoint_node')
        # Drone state
        self.state = State()
        # Instantiate setpoints
        self.sp = PoseStamped()

        # initial values
        self.cur_pose = PoseStamped()
        self.cur_vel = TwistStamped()
        self.acc = Accel()
        self.imu = Imu()
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0
        self.desAngPos = Vector3()
        self.curAngPos = Vector3()
        self.ALT_SP = 2.0
        self.sp.pose.position.z = self.ALT_SP
        self.local_pos = Point(x=0.0, y=0.0, z=self.ALT_SP)
        self.local_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)

        self.att_cmd = AttitudeTarget()

        # controller gains and params (preserved)
        self.Kp0 = np.array([1.0, 1.0, 1.0])
        self.Kp1 = np.array([2.0, 2.0, 1.0])
        self.Lam = np.array([2.0, 2.0, 4.5])
        self.Phi = np.array([1.5, 1.5, 1.2])
        self.M = 0.45
        self.alpha_0 = np.array([1, 1, 1])
        self.alpha_1 = np.array([3, 3, 3])
        self.alpha_m = 0.05
        self.v = 0.1

        self.norm_thrust_const = 0.08
        self.max_th = 20.0
        self.max_throttle = 0.96
        self.gravity = np.array([0, 0, 9.81])

        # use node clock time as previous time (seconds)
        self.pre_time = self.get_clock().now().nanoseconds / 1e9

        # Publishers (QoS depth 10)
        qos_profile = 10
        qos_profile_topic = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.att_pub = self.create_publisher(AttitudeTarget, '/mavros/setpoint_raw/attitude', qos_profile)
        self.sp_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile)
        self.deltau_pub = self.create_publisher(Vector3Stamped, '/deltau', qos_profile)

        # Subscribers
        self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile)
        self.create_subscription(Odometry, '/mavros/local_position/odom', self.odom_cb, qos_profile_topic)
        self.create_subscription(Imu, '/mavros/imu/data', self.acc_cb, qos_profile_topic)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos_profile_topic)
        self.create_subscription(PoseStamped, '/new_pose', self.new_pose_cb, qos_profile)
        self.create_subscription(Mdjt, '/command/trajectory', self.multi_dof_cb, qos_profile)

        # internal flags
        self.armed = False

        # timer for publishing attitude & thrust at 30 Hz
        self._period = 1.0 / 30.0
        self._timer = self.create_timer(self._period, self.timer_callback)

        self.get_logger().info("Controller node started")

    # ---------- callbacks ----------
    def state_cb(self, msg: State):
        self.state = msg

    def pos_cb(self, msg: PoseStamped):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z
        self.local_quat[0] = msg.pose.orientation.x
        self.local_quat[1] = msg.pose.orientation.y
        self.local_quat[2] = msg.pose.orientation.z
        self.local_quat[3] = msg.pose.orientation.w

    def odom_cb(self, msg: Odometry):
        self.cur_pose.pose.position.x = msg.pose.pose.position.x
        self.cur_pose.pose.position.y = msg.pose.pose.position.y
        self.cur_pose.pose.position.z = msg.pose.pose.position.z

        self.cur_pose.pose.orientation.w = msg.pose.pose.orientation.w
        self.cur_pose.pose.orientation.x = msg.pose.pose.orientation.x
        self.cur_pose.pose.orientation.y = msg.pose.pose.orientation.y
        self.cur_pose.pose.orientation.z = msg.pose.pose.orientation.z

        self.cur_vel.twist.linear.x = msg.twist.twist.linear.x
        self.cur_vel.twist.linear.y = msg.twist.twist.linear.y
        self.cur_vel.twist.linear.z = msg.twist.twist.linear.z

        self.cur_vel.twist.angular.x = msg.twist.twist.angular.x
        self.cur_vel.twist.angular.y = msg.twist.twist.angular.y
        self.cur_vel.twist.angular.z = msg.twist.twist.angular.z

    def acc_cb(self, msg: Imu):
        # copy into internal imu structure
        self.imu.orientation = msg.orientation
        self.imu.angular_velocity = msg.angular_velocity
        self.imu.linear_acceleration = msg.linear_acceleration

    def acc_ang(self, msg: Accel):
        self.acc.linear_acc.x = msg.linear.x
        self.acc.linear_acc.y = msg.linear.y
        self.acc.linear_acc.z = msg.linear.z

        self.acc.angular_acc.x = msg.angular.x
        self.acc.angular_acc.y = msg.angular.y
        self.acc.angular_acc.z = msg.angular.z

    def new_pose_cb(self, msg: PoseStamped):
        if (self.sp.pose.position != msg.pose.position):
            self.get_logger().info("New pose received")
        self.sp.pose.position = msg.pose.position
        self.sp.pose.orientation = msg.pose.orientation

    def multi_dof_cb(self, msg: Mdjt):
        if not msg.points:
            return
        pt = msg.points[0]
        # note: transforms/velocities/accelerations may be length >0 or missing depending on publisher
        try:
            t = pt.transforms[0]
            self.sp.pose.position.x = t.translation.x
            self.sp.pose.position.y = t.translation.y
            self.sp.pose.position.z = t.translation.z
        except Exception:
            pass

        try:
            v = pt.velocities[0].linear
            self.desVel = np.array([v.x, v.y, v.z])
        except Exception:
            pass

        try:
            a = pt.accelerations[0].linear
            self.desAcc = np.array([a.x, a.y, a.z])
        except Exception:
            pass

        try:
            av = pt.velocities[0].angular
            self.desAngVel = np.array([av.x, av.y, av.z])
        except Exception:
            pass

        try:
            aa = pt.accelerations[0].angular
            self.desAngAcc = np.array([aa.x, aa.y, aa.z])
        except Exception:
            pass

    # ---------- helpers ----------
    def vector2arrays(self, vector):
        return np.array([vector.x, vector.y, vector.z])

    def vector3arrays(self, vector):
        # note: in your ROS1 code this used x,y,z,w -> kept with same name
        return np.array([vector.x, vector.y, vector.z, getattr(vector, 'w', 0.0)])

    def array2vector3(self, array, vector):
        vector.x = float(array[0])
        vector.y = float(array[1])
        vector.z = float(array[2])

    def array2vector4(self, array, vector):
        vector.x = float(array[0])
        vector.y = float(array[1])
        vector.z = float(array[2])
        vector.w = float(array[3])

    def sigmoid(self, s, v):
        if abs(s) > v:
            return s / abs(s)
        else:
            return s / v

    def th_des(self):
        # compute dt from clock
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.pre_time
        self.pre_time = self.pre_time + dt
        if dt > 0.04:
            dt = 0.04

        curPos = self.vector2arrays(self.cur_pose.pose.position)
        desPos = self.vector2arrays(self.sp.pose.position)
        curVel = self.vector2arrays(self.cur_vel.twist.linear)

        curor = self.vector3arrays(self.cur_pose.pose.orientation)
        curAcc = self.vector2arrays(self.imu.linear_acceleration)

        errPos = curPos - desPos
        errVel = curVel - self.desVel
        sv = errVel + np.multiply(self.Phi, errPos)

        if self.armed:
            self.Kp0 += (sv - np.multiply(self.alpha_0, self.Kp0)) * dt
            self.Kp1 += (sv - np.multiply(self.alpha_1, self.Kp1)) * dt
            self.Kp0 = np.maximum(self.Kp0, 0.0001 * np.ones(3))
            self.Kp1 = np.maximum(self.Kp1, 0.0001 * np.ones(3))
            self.M += (-sv[2] - self.alpha_m * self.M) * dt
            self.M = np.maximum(self.M, 0.1)

        Rho = self.Kp0 + self.Kp1 * errPos

        delTau = np.zeros(3)
        delTau[0] = Rho[0] * self.sigmoid(sv[0], self.v)
        delTau[1] = Rho[1] * self.sigmoid(sv[1], self.v)
        delTau[2] = Rho[2] * self.sigmoid(sv[2], self.v)

        des_th = (
            -np.multiply(self.Lam, sv)
            + self.M * self.gravity)

        if np.linalg.norm(des_th) > self.max_th:
            des_th = (self.max_th / np.linalg.norm(des_th)) * des_th

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(des_th[0])
        msg.vector.y = float(des_th[1])
        msg.vector.z = float(des_th[2])
        self.deltau_pub.publish(msg)

        return des_th

    def acc2quat(self, des_th, des_yaw):
        des_th = des_th[0:3]
        proj_xb_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        if np.linalg.norm(des_th) == 0.0:
            zb_des = np.array([0.0, 0.0, 1.0])
        else:
            zb_des = des_th / np.linalg.norm(des_th)

        # cross products
        cross1 = np.cross(zb_des, proj_xb_des)
        # protect against zero cross product
        if np.linalg.norm(cross1) == 0:
            # choose arbitrary orthogonal yb_des
            yb_des = np.array([0.0, 1.0, 0.0])
        else:
            yb_des = cross1 / np.linalg.norm(cross1)

        cross2 = np.cross(yb_des, zb_des)
        if np.linalg.norm(cross2) == 0:
            xb_des = np.array([1.0, 0.0, 0.0])
        else:
            xb_des = cross2 / np.linalg.norm(cross2)

        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat

    def geo_con(self):
        des_th = self.th_des()
        des_th = des_th[0:3]
        r_des = self.acc2quat(des_th, 0.0)

        # Build 4x4 transform for quaternion extraction
        rot_44 = np.vstack((np.hstack((r_des, np.array([[0.0, 0.0, 0.0]]).T)), np.array([[0.0, 0.0, 0.0, 1.0]])))
        if tf_trans is not None and hasattr(tf_trans, 'quaternion_from_matrix'):
            quat_des = tf_trans.quaternion_from_matrix(rot_44)
        else:
            # fallback simple conversion (not robust). Recommend installing tf_transformations.
            # Construct quaternion from rotation matrix manually (approx). Better to install tf_transformations.
            m = r_des
            tr = m[0, 0] + m[1, 1] + m[2, 2]
            if tr > 0:
                S = math.sqrt(tr + 1.0) * 2.0
                qw = 0.25 * S
                qx = (m[2, 1] - m[1, 2]) / S
                qy = (m[0, 2] - m[2, 0]) / S
                qz = (m[1, 0] - m[0, 1]) / S
                quat_des = [qx, qy, qz, qw]
            else:
                quat_des = [0.0, 0.0, 0.0, 1.0]

        zb = r_des[:, 2]
        thrust = self.norm_thrust_const * float(np.dot(des_th, zb))
        thrust = float(np.maximum(0.0, np.minimum(thrust, self.max_throttle)))

        now = self.get_clock().now().to_msg()  # builtin ROS2 time msg

        self.att_cmd.header.stamp = now
        self.att_cmd.orientation.x = float(quat_des[0])
        self.att_cmd.orientation.y = float(quat_des[1])
        self.att_cmd.orientation.z = float(quat_des[2])
        self.att_cmd.orientation.w = float(quat_des[3])
        self.att_cmd.thrust = thrust
        self.att_cmd.type_mask = 7


    def pub_att(self):
        self.geo_con()
        self.att_pub.publish(self.att_cmd)

    # timer called at 30 Hz
    def timer_callback(self):
        # publish position setpoints a few times as in original script
        # self.sp_pub.publish(self.sp)
        # publish attitude each main loop iteration
        # set self.armed based on state; keep internal flag too
        self.armed = getattr(self.state, 'armed', self.armed)
        self.pub_att()


def main(argv=None):
    rclpy.init(args=argv)
    node = Controller()
    modes = FCUModesROS2(node)

    # arm loop similar to original — try until armed flag set
    node.get_logger().info("ARMING")
    try_count = 0
    while rclpy.ok() and not getattr(node.state, 'armed', False) and try_count < 10:
        modes.set_arm(True)
        node.armed = True
        try_count += 1
        # small sleep
        rclpy.spin_once(node, timeout_sec=0.1)

    # publish initial setpoints a few times (like original)
    for _ in range(20):
        node.sp_pub.publish(node.sp)
        rclpy.spin_once(node, timeout_sec=1.0 / 30.0)

    # set offboard mode
    modes.set_mode('OFFBOARD')
    node.get_logger().info("OFFBOARD")

    try:
        # keep spinning until shutdown
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
