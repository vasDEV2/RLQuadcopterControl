#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
# import jax.numpy as jnp
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

if not hasattr(np, "maximum_sctype"):
    def _maximum_sctype(t):
        return np.float64
    np.maximum_sctype = _maximum_sctype

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Imu

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Vector3, Vector3Stamped
import tf_transformations as transformations
from tf_transformations import *

# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import CommandTOL, CommandBool, SetMode
from nav_msgs.msg import *
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from std_msgs.msg import Float32
# from gazebo_msgs.msg import ModelStates
# from msg_check.msg import PlotDataMsg

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# from diffusion.diffusion_inference import Policy


class fcuModes:
    """
    Minimal ROS2-compatible conversion of your fcuModes service wrappers.
    This class requires a rclpy.node.Node instance to create clients.
    """

    def __init__(self, node: Node):
        self.node = node
        self._cbg = ReentrantCallbackGroup()

    def _call_service_sync(self, client, req):
        # Wait for service
        if not client.wait_for_service(timeout_sec=5.0):
            # try waiting longer if not available
            self.node.get_logger().warn(f"Service {client.srv_name} not available yet.")
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Service {client.srv_name} not available.")

        fut = client.call_async(req)
        # spin until result is ready
        rclpy.spin_until_future_complete(self.node, fut)
        if fut.result() is None:
            raise RuntimeError(f"Service call {client.srv_name} failed or returned None.")
        return fut.result()

    def setTakeoff(self):
        client = self.node.create_client(CommandTOL, "mavros/cmd/takeoff", callback_group=self._cbg)
        req = CommandTOL.Request()
        req.altitude = 3.0
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"Service takeoff call failed: {e}")

    def setArm(self):
        client = self.node.create_client(CommandBool, "mavros/cmd/arming", callback_group=self._cbg)
        req = CommandBool.Request()
        req.value = True
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"Service arming call failed: {e}")

    def setDisarm(self):
        client = self.node.create_client(CommandBool, "mavros/cmd/arming", callback_group=self._cbg)
        req = CommandBool.Request()
        req.value = False
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"Service disarming call failed: {e}")

    def setStabilizedMode(self):
        client = self.node.create_client(SetMode, "mavros/set_mode", callback_group=self._cbg)
        req = SetMode.Request()
        req.custom_mode = "STABILIZED"
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"service set_mode call failed: {e}. Stabilized Mode could not be set.")

    def setOffboardMode(self):
        client = self.node.create_client(SetMode, "mavros/set_mode", callback_group=self._cbg)
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"service set_mode call failed: {e}. Offboard Mode could not be set.")

    def setAltitudeMode(self):
        client = self.node.create_client(SetMode, "mavros/set_mode", callback_group=self._cbg)
        req = SetMode.Request()
        req.custom_mode = "ALTCTL"
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"service set_mode call failed: {e}. Altitude Mode could not be set.")

    def setPositionMode(self):
        client = self.node.create_client(SetMode, "mavros/set_mode", callback_group=self._cbg)
        req = SetMode.Request()
        req.custom_mode = "POSCTL"
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"service set_mode call failed: {e}. Position Mode could not be set.")

    def setAutoLandMode(self):
        client = self.node.create_client(SetMode, "mavros/set_mode", callback_group=self._cbg)
        req = SetMode.Request()
        req.custom_mode = "AUTO.LAND"
        try:
            self._call_service_sync(client, req)
        except Exception as e:
            self.node.get_logger().error(f"service set_mode call failed: {e}. Autoland Mode could not be set.")


class Controller(Node):
    # initialization method
    def __init__(self):
        super().__init__("setpoint_node")

        self.state = State()
        self.sp = PoseStamped()
        self.yaw_angle = Float32()
        self.yaw_angle.data = 0.0

        self.imu = Imu()

        self.sp.pose.position.x = 0.0
        self.sp.pose.position.y = 0.0
        self.sp.pose.position.z = 2.0

        self.local_pos = PoseStamped()
        self.local_vel = TwistStamped()

        self.local_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)

        self.att_cmd = AttitudeTarget()
        self.thrust_cmd = Thrust()

        # Gains
        self.Lam = np.array([2.0, 2.0, 8.0])
        self.Phi = np.array([1.5, 1.5, 1.2])
        self.M_bar = 1.0
        self.Kp0, self.alpha_0, self.v = (
            np.array([0.1, 0.1, 0.1]),
            np.array([1, 1, 1]),
            0.1,
        )

        self.norm_thrust_const =  0.08 #0.056
        self.max_th = 12.0
        self.max_throttle = 0.96
        self.gravity = np.array([0, 0, 9.8])
        self.pre_time = self.get_clock().now().nanoseconds * 1e-9

        self.armed = False
        self.command = AttitudeTarget()
        self.collective_thrust = 0.0

        # Publishers (ROS2 style)
        # self.att_pub = self.create_publisher(PoseStamped, "mavros/setpoint_attitude/attitude", 10)
        # self.thrust_pub = self.create_publisher(Thrust, "mavros/setpoint_attitude/thrust", 10)
        self.att_pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)
        self.deltau_pub = self.create_publisher(Vector3Stamped, '/deltau', 10)

        # Observations / last values
        self.obs = None
        self.last_tau = np.array([0, 0, 0])
        self.last_vel = np.array([0, 0, 0])
        self.last_accel = np.array([0, 0, 0])

        # Diffusion policy: unchanged
        # self.policy = Policy(config="experiment.yaml", use_ema=True, rng=42)
        # self.policy(jnp.ones((15,), dtype=jnp.float32))  # Warm up the JAX model

    # callbacks to populate state
    # def base_link_pos(self, msg: ModelStates):
    #     try:
    #         idx = msg.name.index("iris")
    #     except ValueError:
    #         # if model not found, just return
    #         return

    #     iris_pose = msg.pose[idx]
    #     iris_twist = msg.twist[idx]

    #     self.local_pos.pose.position.x = iris_pose.position.x
    #     self.local_pos.pose.position.y = iris_pose.position.y
    #     self.local_pos.pose.position.z = iris_pose.position.z

    #     self.local_pos.pose.orientation.x = iris_pose.orientation.x
    #     self.local_pos.pose.orientation.y = iris_pose.orientation.y
    #     self.local_pos.pose.orientation.z = iris_pose.orientation.z
    #     self.local_pos.pose.orientation.w = iris_pose.orientation.w

    #     self.local_vel.twist.linear.x = iris_twist.linear.x
    #     self.local_vel.twist.linear.y = iris_twist.linear.y
    #     self.local_vel.twist.l/home/shivansh/DroneDiffusion_jax/scripts/drone_controller.pyinear.z = iris_twist.linear.z

    #     self.local_vel.twist.angular.x = iris_twist.angular.x
    #     self.local_vel.twist.angular.y = iris_twist.angular.y
    #     self.local_vel.twist.angular.z = iris_twist.angular.z

    def posCb(self, msg: PoseStamped):
        self.local_pos.pose.position.x = msg.pose.position.x
        self.local_pos.pose.position.y = msg.pose.position.y
        self.local_pos.pose.position.z = msg.pose.position.z

        self.local_pos.pose.orientation.x = msg.pose.orientation.x
        self.local_pos.pose.orientation.y = msg.pose.orientation.y
        self.local_pos.pose.orientation.z = msg.pose.orientation.z
        self.local_pos.pose.orientation.w = msg.pose.orientation.w

    def velCb(self, msg: TwistStamped):
        self.local_vel.twist.linear.x = msg.twist.linear.x
        self.local_vel.twist.linear.y = msg.twist.linear.y
        self.local_vel.twist.linear.z = msg.twist.linear.z

        self.local_vel.twist.angular.x = msg.twist.angular.x
        self.local_vel.twist.angular.y = msg.twist.angular.y
        self.local_vel.twist.angular.z = msg.twist.angular.z

    def multiDoFCb(self, msg: Mdjt):
        if not msg.points:
            return
        pt = msg.points[0]
        if not pt.transforms:
            return
        self.sp.pose.position.x = pt.transforms[0].translation.x
        self.sp.pose.position.y = pt.transforms[0].translation.y
        self.sp.pose.position.z = pt.transforms[0].translation.z
        self.desVel = np.array(
            [
                pt.velocities[0].linear.x,
                pt.velocities[0].linear.y,
                pt.velocities[0].linear.z,
            ]
        )
        self.desAcc = np.array(
            [
                pt.accelerations[0].linear.x,
                pt.accelerations[0].linear.y,
                pt.accelerations[0].linear.z,
            ]
        )

    def stateCb(self, msg: State):
        self.state = msg

    def updateSp(self):
        self.sp.pose.position.x = self.local_pos.pose.position.x
        self.sp.pose.position.y = self.local_pos.pose.position.y
        self.sp.pose.position.z = self.local_pos.pose.position.z

    def accCB(self, msg: Imu):
        self.imu.orientation.w = msg.orientation.w
        self.imu.orientation.x = msg.orientation.x
        self.imu.orientation.y = msg.orientation.y
        self.imu.orientation.z = msg.orientation.z

        self.imu.angular_velocity.x = msg.angular_velocity.x
        self.imu.angular_velocity.y = msg.angular_velocity.y
        self.imu.angular_velocity.z = msg.angular_velocity.z

        self.imu.linear_acceleration.x = msg.linear_acceleration.x
        self.imu.linear_acceleration.y = msg.linear_acceleration.y
        self.imu.linear_acceleration.z = msg.linear_acceleration.z

    def newPoseCB(self, msg: PoseStamped):
        if self.sp.pose.position != msg.pose.position:
            self.get_logger().info("New pose received")
        self.sp.pose.position.x = msg.pose.position.x
        self.sp.pose.position.y = msg.pose.position.y
        self.sp.pose.position.z = msg.pose.position.z

        self.sp.pose.orientation.x = msg.pose.orientation.x
        self.sp.pose.orientation.y = msg.pose.orientation.y
        self.sp.pose.orientation.z = msg.pose.orientation.z
        self.sp.pose.orientation.w = msg.pose.orientation.w

    def yawAngle(self, msg: Float32):
        self.yaw_angle.data = msg.data

    def vector2Arrays(self, vector):
        return np.array([vector.x, vector.y, vector.z])

    def vector3Arrays(self, vector):
        # keep same name, original returned 4-length (with .w) but used only for quaternions earlier
        return np.array([vector.x, vector.y, vector.z, getattr(vector, "w", 0.0)])

    def array2Vector3(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]

    def array2Vector4(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]
        vector.w = array[3]

    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return s / np.absolute(s)
        else:
            return s / v

    def get_obs(self):
        curPos = self.vector2Arrays(self.local_pos.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.gravity

        orientation_q = self.local_pos.pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (roll_curr, pitch_curr, yaw_curr) = euler_from_quaternion(orientation_list)

        accel = curAcc
        obs = np.array(
            [
                curPos[0],
                curPos[1],
                curPos[2],
                curVel[0],
                curVel[1],
                curVel[2],
                curAcc[0],
                curAcc[1],
                curAcc[2],
                roll_curr,
                pitch_curr,
                yaw_curr,
                self.last_tau[0],
                self.last_tau[1],
                self.last_tau[2],
            ],
            dtype=np.float32,
        )

        return obs, accel

    def get_action(self, curr_act):
        self.currN = curr_act

    def a_des(self):
        _, accel = self.get_obs()

        dt = self.get_clock().now().nanoseconds * 1e-9 - self.pre_time
        self.pre_time = self.pre_time + dt

        if dt > 0.01:
            dt = 0.01

        curPos = self.vector2Arrays(self.local_pos.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        # print(curVel)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.gravity

        errPos = curPos - desPos
        errVel = curVel - self.desVel
        errAcc = curAcc - self.desAcc

        sv = errVel + np.multiply(self.Phi, errPos)
        
        if self.armed:
            self.Kp0 = (np.linalg.norm(sv) - np.multiply(self.alpha_0, self.Kp0)) * dt
            # self.Kp0 += (sv - np.multiply(self.alpha_0, self.Kp0)) * dt
            self.Kp0 = np.maximum(self.Kp0, 0.0001 * np.ones(3))
        Rho = self.Kp0
        delTau = np.zeros(3)
        delTau[0] = Rho[0] * self.sigmoid(sv[0], self.v)
        delTau[1] = Rho[1] * self.sigmoid(sv[1], self.v)
        delTau[2] = Rho[2] * self.sigmoid(sv[2], self.v)

        des_a = (
            -np.multiply(self.Lam, sv)
            + self.M_bar * (self.desAcc + self.gravity - np.multiply(self.Phi, errVel))
            #+ self.currN
            - delTau
        )

        # self.get_logger().debug(f"N   : {self.currN}")
        print(f"errPos {errPos}")
        self.get_logger().debug("------------")

        if np.linalg.norm(des_a) > self.max_th:
            des_a = (self.max_th / np.linalg.norm(des_a)) * des_a

        self.last_tau = des_a
        self.last_vel = curVel
        self.last_accel = accel

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(des_a[0])
        msg.vector.y = float(des_a[1])
        msg.vector.z = float(des_a[2])
        self.deltau_pub.publish(msg)

        return des_a

    def geo_con_(self):
        des_a = self.a_des()

        pose = transformations.quaternion_matrix(
            np.array(
                [
                    self.local_pos.pose.orientation.x,
                    self.local_pos.pose.orientation.y,
                    self.local_pos.pose.orientation.z,
                    self.local_pos.pose.orientation.w,
                ]
            )
        )  # 4*4 matrix
        pose_temp1 = np.delete(pose, -1, axis=1)
        rot_curr = np.delete(pose_temp1, -1, axis=0)  # 3*3 current rotation matrix
        zb_curr = rot_curr[:, 2]
        thrust = self.norm_thrust_const * des_a.dot(zb_curr)
        self.collective_thrust = np.maximum(0.0, np.minimum(thrust, self.max_throttle))

        # Desired Euler orientation and Desired Rotation matrix
        rot_des = self.acc2quat(des_a, 0)  # desired yaw = 0
        rot_44 = np.vstack(
            (np.hstack((rot_des, np.array([[0, 0, 0]]).T)), np.array([[0, 0, 0, 1]]))
        )
        quat_des = quaternion_from_matrix(rot_44)

        now = self.get_clock().now().to_msg()
        self.att_cmd.header.stamp = now
        self.att_cmd.orientation.x = quat_des[0]
        self.att_cmd.orientation.y = quat_des[1]
        self.att_cmd.orientation.z = quat_des[2]
        self.att_cmd.orientation.w = quat_des[3]
        self.att_cmd.thrust = float(self.collective_thrust)
        self.att_cmd.type_mask = 7

    def acc2quat(self, des_a, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        if np.linalg.norm(des_a) == 0.0:
            zb_des = np.array([0, 0, 1])
        else:
            zb_des = des_a / np.linalg.norm(des_a)
        # ensure cross products are safe (avoid zero division)
        cross_y = np.cross(zb_des, proj_xb_des)
        if np.linalg.norm(cross_y) == 0:
            # fallback orthogonal vector
            if abs(zb_des[2]) < 0.9:
                cross_y = np.cross(zb_des, np.array([0, 0, 1.0]))
            else:
                cross_y = np.cross(zb_des, np.array([0, 1.0, 0]))
        yb_des = cross_y / np.linalg.norm(cross_y)
        xb_des = np.cross(yb_des, zb_des)
        xb_des = xb_des / np.linalg.norm(xb_des)

        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat

    def pub_att(self):
        self.geo_con_()
        # self.thrust_pub.publish(self.thrust_cmd)
        self.att_pub.publish(self.att_cmd)


def main(argv):
    rclpy.init(args=argv)
    node = Controller()
    modes = fcuModes(node)
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10
    )

    # create subscriptions (mimic original ROS1 behavior)
    node.create_subscription(State, "/mavros/state", node.stateCb, 10)
    # node.create_subscription(ModelStates, "/gazebo/model_states", node.base_link_pos, 10)
    node.create_subscription(Imu, "/mavros/imu/data", node.accCB, qos)
    node.create_subscription(PoseStamped, "/mavros/local_position/pose", node.posCb, qos)
    node.create_subscription(TwistStamped, "/mavros/local_position/velocity_local", node.velCb, qos)
    node.create_subscription(Mdjt, "/command/trajectory", node.multiDoFCb, 10)
    node.create_subscription(PoseStamped, "new_pose", node.newPoseCB, 10)
    node.create_subscription(Float32, "yaw_in_deg", node.yawAngle, 10)

    # setpoint position publisher (used in arming phase)
    sp_pub = node.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)

    node.get_logger().info("ARMING")

    # attempt to arm until armed; mimics original loop (note: may be aggressive)
    while not node.state.armed and rclpy.ok():
        try:
            modes.setArm()
            node.armed = True
        except Exception as e:
            node.get_logger().warn(f"Arm attempt failed: {e}")
        # process callbacks
        # rclpy.spin_once(node, timeout_sec=0.1)

    node.armed = True
    k = 0

    # publish a few setpoints before switching to offboard (as originally)
    # rate = node.create_rate(30)
    while k < 20 and rclpy.ok():
        sp_pub.publish(node.sp)
        rclpy.spin_once(node, timeout_sec=1.0 / 30.0)
        k += 1

    try:
        modes.setOffboardMode()
        node.get_logger().info("---------")
        node.get_logger().info("OFFBOARD")
        node.get_logger().info("---------")
    except Exception as e:
        node.get_logger().error(f"Failed to set offboard: {e}")

    # Main loop
    i = 0
    PLANNING_BUDGET = 8

    try:
        while rclpy.ok():
            # if i % PLANNING_BUDGET == 0:
                # obs = node.get_obs()[0]
                # action = node.policy(obs)

            for j in range(PLANNING_BUDGET):
                # curr_act = action[j]
                # node.get_action(curr_act)

                node.pub_att()
                # allow callbacks to run
                rclpy.spin_once(node, timeout_sec=0.0)

            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
