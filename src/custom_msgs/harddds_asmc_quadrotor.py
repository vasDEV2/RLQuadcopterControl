#!/usr/bin/env python3
import sys
# import jax.numpy as jnp
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped, Vector3Stamped
import tf_transformations as transformations
from tf_transformations import *

from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from std_msgs.msg import Float32
# from gazebo_msgs.msg import ModelStates
# from msg_check.msg import PlotDataMsg

from px4_msgs.msg import VehicleCommand, VehicleOdometry, VehicleThrustSetpoint, VehicleTorqueSetpoint, OffboardControlMode, VehicleRatesSetpoint, VehicleStatus, VehicleAttitudeSetpoint, VehicleAcceleration


from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# from inference import TimeXerInferRaw, RealTimeInference
from collections import deque
import csv


class OneStepAheadInference:
    def __init__(self, inferencer, enc_in, c_out, mass=1.0):
        self.inferencer = inferencer
        self.seq_len = inferencer.seq_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.mass = mass
        
        self.state_buffer = deque(maxlen=self.seq_len)
        self.H_actual_buffer = deque(maxlen=self.seq_len)
        
        self.step_count = 0
        self.is_warmed_up = False
        
        # Store prediction for next step
        self.prediction_for_next_step = np.zeros(c_out, dtype=np.float32)
        self.prediction_for_current_step = np.zeros(c_out, dtype=np.float32)
        
        print(f"✓ One-Step Ahead Inference initialized")
        print(f"✓ Warmup required: {self.seq_len} steps")
        print(f"✓ Using ACTUAL H measurements in history")
        print(f"✓ Predicting ONE step ahead EVERY iteration")
        
    def add_step_and_predict(self, state_15d, tau_actual, accel_actual):
        self.prediction_for_current_step = self.prediction_for_next_step.copy()
        
        H_actual = tau_actual - self.mass * accel_actual
        state_18d = np.concatenate([state_15d, H_actual])
        
        self.state_buffer.append(state_18d)
        self.H_actual_buffer.append(H_actual)
        
        self.step_count += 1
        
        if self.step_count >= self.seq_len:
            if not self.is_warmed_up:
                print(f"✓ Warmup complete!")
                # Print buffer stats after warmup
                state_arr = np.array(self.state_buffer)
                H_arr = np.array(self.H_actual_buffer)
                print(f"Buffer state range: [{state_arr.min():.3f}, {state_arr.max():.3f}]")
                print(f"Buffer H range: [{H_arr.min():.3f}, {H_arr.max():.3f}]")
                print(f"Buffer state mean: {state_arr.mean():.3f}, std: {state_arr.std():.3f}")
                print(f"Buffer H mean: {H_arr.mean():.3f}, std: {H_arr.std():.3f}")
                
            self.is_warmed_up = True
            
            state_window = np.array(self.state_buffer)
            H_window = np.array(self.H_actual_buffer)
                        
            self.prediction_for_next_step = self.inferencer.predict_once(
                state_window, H_window
            )
                        
            if self.step_count % 100 == 0:
                print(f"Step {self.step_count}: H_actual={H_actual}, H_pred_next={self.prediction_for_next_step}")
        else:
            self.prediction_for_next_step = np.zeros(self.c_out, dtype=np.float32)
            if self.step_count % 5 == 0:
                print(f"Warmup: {self.step_count}/{self.seq_len}")

    def get_current_prediction(self):
        if not self.is_warmed_up:
            return np.zeros(self.c_out, dtype=np.float32)
        return self.prediction_for_current_step
    
    def reset(self):
        self.state_buffer.clear()
        self.H_actual_buffer.clear()
        self.step_count = 0
        self.is_warmed_up = False
        self.prediction_for_next_step = np.zeros(self.c_out, dtype=np.float32)
        self.prediction_for_current_step = np.zeros(self.c_out, dtype=np.float32)

class fcuModes:
    def __init__(self, node: Node):
        self.node = node
        self._cbg = ReentrantCallbackGroup()
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.vehicle_command_publisher = self.node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.node.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.node.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.node.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.node.get_logger().info("Switching to land mode")

class Controller(Node):
    # initialization method
    def __init__(self):
        super().__init__("setpoint_node")

        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth = 1
        )

        self.state = VehicleStatus()
        self.sp = PoseStamped()
        self.yaw_angle = Float32()
        self.yaw_angle.data = 0.0

        self.sp.pose.position.y = 0.0
        self.sp.pose.position.x = 0.0
        self.sp.pose.position.z = 1.0

        self.curr_position = np.zeros(3, dtype=float)
        self.curr_orientation = np.zeros(4, dtype=float)
        self.curr_linear_vel = np.zeros(3, dtype=float)
        self.curr_angular_vel = np.zeros(3, dtype=float)
        self.curr_linear_acc = np.zeros(3, dtype=float)

        self.local_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desAngVel = np.zeros(3)
        self.desAngAcc = np.zeros(3)

        self.offboard_setpoint_counter = 0

        # Gains
        self.Lam = np.array([2.0, 2.0, 8.0])
        self.Phi = np.array([1.5, 1.5, 1.2])
        self.M_bar = 1.0
        self.Kp0, self.alpha_0, self.v = (
            np.array([0.1, 0.1, 0.1]),
            np.array([1, 1, 1]),
            0.1,
        )

        self.norm_thrust_const = 0.062 # For manipulator: 0.052, Only Drone: 0.036
        self.max_th = 12.0
        self.max_throttle = 0.96
        self.gravity = np.array([0, 0, 9.8])
        self.pre_time = self.get_clock().now().nanoseconds * 1e-9

        # Hardware 1 (14-01-2026)
        # self.Lam = np.array([2.0, 2.0, 5.5])
        # self.Phi = np.array([1.5, 1.5, 1.2])
        # self.M_bar = 0.45
        # self.Kp0, self.alpha_0, self.v = (
        #     np.array([0.1, 0.1, 0.1]),
        #     np.array([1, 1, 1]),
        #     0.1,
        # )

        # self.norm_thrust_const = 0.08
        # self.max_th = 5.0
        # self.max_throttle = 0.96
        # self.gravity = np.array([0, 0, 9.8])
        # self.pre_time = self.get_clock().now().nanoseconds * 1e-9

        self.armed = False
        self.collective_thrust = 0.0
        self.desH = np.array([0.0, 0.0, 0.0])
        self.predH = np.array([0.0, 0.0, 0.0])

        self.des_yaw = None

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', 10)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', 10)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', 10)
        self.att_sp_pub = self.create_publisher(VehicleAttitudeSetpoint,'/fmu/in/vehicle_attitude_setpoint', 10)

        self.deltau_pub = self.create_publisher(Vector3Stamped, '/deltau', 10)
        self.acc_pub = self.create_publisher(Vector3Stamped, '/acc', 10)
        self.desH_pub = self.create_publisher(Vector3Stamped, '/H_actual', 10)
        self.predH_pub = self.create_publisher(Vector3Stamped, '/H_pred', 10)

        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth = 1
        )

        self.create_subscription(VehicleAcceleration, '/fmu/out/vehicle_acceleration', self.accel_callback, qos_profile=sim_qos_profile)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_sub_callback, qos_profile=sim_qos_profile)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.stateCb, qos_profile=sim_qos_profile)
        self.create_subscription(Mdjt, "command/trajectory", self.multiDoFCb, 10)

        # Observations / last values
        self.obs = None
        self.last_tau = np.array([0, 0, 0])
        self.last_vel = np.array([0, 0, 0])
        self.last_accel = np.array([0, 0, 0])

        # inference = TimeXerInferRaw(
        #     config_path="/home/control-lab/AMTimeXer/config/asmc_v2_seq10_pred2.yaml",
        #     ckpt_path="/home/control-lab/AMTimeXer/checkpoints/hard/long_term_forecast_asmc_v2_TimeXer_V2_custom_ftM_sl10_ll8_pl2_dm512_nh8_el4_dl2_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_1/checkpoint.pth",
        #     device="cuda"
        # )

        # self.infer = OneStepAheadInference(
        #     inference, 
        #     enc_in=15, 
        #     c_out=3,
        #     mass=1.0
        # )

        # self.i = 0

        # print("Starting real-time inference...")
        # print(f"Warmup: Need {self.infer.seq_len} steps before predictions are reliable")

        self.control_timer = self.create_timer(1/70.0, self.control_loop)

    def control_loop(self):
        if self.state.nav_state == 14:
            self.offboard_signal()
            self.pub_att()
            print("In OFFBOARD mode and armed")
        else:
            pass
 
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
        q = pt.transforms[0].rotation
        self.desRPY = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.des_yaw = self.desRPY[2]


    def stateCb(self, msg: VehicleStatus):
        self.state = msg
    
    def accel_callback(self, msg: VehicleAcceleration):
        self.curr_linear_acc[0] = msg.xyz[0]
        self.curr_linear_acc[1] = -msg.xyz[1]
        self.curr_linear_acc[2] = -msg.xyz[2]

    def odom_sub_callback(self, msg: VehicleOdometry):
        self.curr_position[0] = msg.position[0]
        self.curr_position[1] = -msg.position[1]
        self.curr_position[2] = -msg.position[2]

        self.curr_orientation[0] = msg.q[1]
        self.curr_orientation[1] = -msg.q[2]
        self.curr_orientation[2] = -msg.q[3]
        self.curr_orientation[3] = msg.q[0]

        self.curr_linear_vel[0] = msg.velocity[0]
        self.curr_linear_vel[1] = -msg.velocity[1]
        self.curr_linear_vel[2] = -msg.velocity[2]
        
        self.curr_angular_vel[0] = msg.angular_velocity[0]
        self.curr_angular_vel[1] = -msg.angular_velocity[1]
        self.curr_angular_vel[2] = -msg.angular_velocity[2]

    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return s / np.absolute(s)
        else:
            return s / v
        
    def vee(self, M):
        return np.array([M[2, 1], M[0, 2], M[1, 0]])
    
    def vector2Arrays(self, vector):
        return np.array([vector.x, vector.y, vector.z])

    def get_obs(self):
        curPos = self.curr_position
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.curr_linear_vel
        curAcc = self.curr_linear_acc

        orientation_q = self.curr_orientation
        orientation_list = [
            orientation_q[0],
            orientation_q[1],
            orientation_q[2],
            orientation_q[3],
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
                # self.predH[0],
                # self.predH[1],
                # self.predH[2],
                # self.desH[0],
                # self.desH[1],
                # self.desH[2],

            ],
            dtype=np.float32,
        )

        return obs

    def get_action(self):
        """Update inference with detailed debugging."""
        obs = self.get_obs()  # 15D
        
        # Compute H_actual for logging
        self.desH = self.last_tau - 1.0 * self.last_accel
        if self.i >= 1000:
        # Add step
            self.infer.add_step_and_predict(
                obs,
                self.last_tau,
                self.last_accel
            )
            self.predH = self.infer.get_current_prediction()
            # self.predH[2] = 0.2 * np.clip(self.predH[2], -2.0, 2.0)
            # self.predH[2] = -self.predH[2]
        
        H_error = self.desH - self.predH

        self.i += 1
            
        print(f"Tau: [{self.last_tau[0]:.3f}, {self.last_tau[1]:.3f}, {self.last_tau[2]:.3f}]")
        print(f"Accel: [{self.last_accel[0]:.3f}, {self.last_accel[1]:.3f}, {self.last_accel[2]:.3f}]")
        print(f"H_actual: [{self.desH[0]:.3f}, {self.desH[1]:.3f}, {self.desH[2]:.3f}]")
        print(f"H_pred:   [{self.predH[0]}, {self.predH[1]}, {self.predH[2]}]")
        print(f"H_error:  [{H_error[0]:.3f}, {H_error[1]:.3f}, {H_error[2]:.3f}] norm={np.linalg.norm(H_error):.3f}")
            
        if np.linalg.norm(self.predH) > 50:
            print(f"WARNING: H prediction is very large!")
        if np.linalg.norm(H_error) > 5:
            print(f"WARNING: H prediction error is large!")
        
        # self.step_counter += 1

    def a_des(self):
        # _, accel = self.get_obs()

        dt = self.get_clock().now().nanoseconds * 1e-9 - self.pre_time
        self.pre_time = self.pre_time + dt

        if dt > 0.01:
            dt = 0.01

        curPos = self.curr_position
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.curr_linear_vel
        curAcc = self.curr_linear_acc

        errPos = curPos - desPos
        errVel = curVel - self.desVel
        # print(errPos)

        sv = errVel + np.multiply(self.Phi, errPos)
        # print(sv)
        
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
            # - self.predH
            - delTau
        )

        if np.linalg.norm(des_a) > self.max_th:
            des_a = (self.max_th / np.linalg.norm(des_a)) * des_a

        self.last_tau = des_a
        # self.last_vel = curVel
        self.last_accel = curAcc

        # self.get_logger().debug(f"N   : {self.currN}")
        print(f"errPos {errPos}") 
        self.get_logger().debug("------------")

        msg1 = Vector3Stamped()
        msg1.header.stamp = self.get_clock().now().to_msg()
        msg1.vector.x = float(des_a[0])
        msg1.vector.y = float(des_a[1])
        msg1.vector.z = float(des_a[2])
        self.deltau_pub.publish(msg1)

        msg4 = Vector3Stamped()
        msg4.header.stamp = self.get_clock().now().to_msg()
        msg4.vector.x = float(curAcc[0])
        msg4.vector.y = float(curAcc[1])
        msg4.vector.z = float(curAcc[2])
        self.acc_pub.publish(msg4)


        msg2 = Vector3Stamped()
        msg2.header.stamp = self.get_clock().now().to_msg()
        msg2.vector.x = float(self.predH[0])
        msg2.vector.y = float(self.predH[1])
        msg2.vector.z = float(self.predH[2])
        self.predH_pub.publish(msg2)

        msg3 = Vector3Stamped()
        msg3.header.stamp = self.get_clock().now().to_msg()
        msg3.vector.x = float(self.desH[0])
        msg3.vector.y = float(self.desH[1])
        msg3.vector.z = float(self.desH[2])
        self.desH_pub.publish(msg3)

        return des_a

    def geo_con_(self):
        des_a = self.a_des()
        _, _, curr_yaw = euler_from_quaternion([self.curr_orientation[0], self.curr_orientation[1], self.curr_orientation[2], self.curr_orientation[3]])
        print(curr_yaw)
        # if self.des_yaw == None:
        curr_yaw = -1.57
            # _, _, curr_yaw = euler_from_quaternion([self.curr_orientation[0], self.curr_orientation[1], self.curr_orientation[2], self.curr_orientation[3]])
        # else:
        #     curr_yaw = self.des_yaw 
        rot_des = self.acc2quat(des_a, curr_yaw)

        pose = transformations.quaternion_matrix(
            np.array(
                [
                    self.curr_orientation[0],
                    self.curr_orientation[1],
                    self.curr_orientation[2],
                    self.curr_orientation[3],
                ]
            )
        )  # 4*4 matrix
        pose_temp1 = np.delete(pose, -1, axis=1)
        rot_curr = np.delete(pose_temp1, -1, axis=0)  # 3*3 current rotation matrix
        zb_curr = rot_curr[:, 2]
        thrust = self.norm_thrust_const * des_a.dot(zb_curr)
        self.collective_thrust = np.maximum(0.0, np.minimum(thrust, self.max_throttle))

        # Desired Euler orientation and Desired Rotation matrix
        rot_44 = np.vstack(
            (np.hstack((rot_des, np.array([[0, 0, 0]]).T)), np.array([[0, 0, 0, 1]]))
        )
        quat_des = quaternion_from_matrix(rot_44)

        att_sp = VehicleAttitudeSetpoint()
        att_sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        att_sp.q_d[0] = quat_des[3]  # w
        att_sp.q_d[1] = quat_des[0]  # x
        att_sp.q_d[2] = -quat_des[1]  # y
        att_sp.q_d[3] = -quat_des[2]  # z
        att_sp.thrust_body[0] = 0.0
        att_sp.thrust_body[1] = 0.0
        att_sp.thrust_body[2] = -float(self.collective_thrust)
        self.att_sp_pub.publish(att_sp)
        
        return rot_des
        
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

    def offboard_signal(self):
        self.offboard_msg = OffboardControlMode()
        self.offboard_msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_msg.thrust_and_torque = False
        self.offboard_msg.body_rate = False
        self.offboard_msg.attitude = True
        self.publisher_offboard_mode.publish(self.offboard_msg)


def main(argv):
    rclpy.init(args=argv)
    node = Controller()
    modes = fcuModes(node)

    node.get_logger().info("Waiting for system to be ready...")

    # Give PX4 some time to initialize before switching modes
    # for _ in range(10):
    #     node.offboard_signal()  # start sending offboard heartbeat
    #     rclpy.spin_once(node, timeout_sec=0.1)

    # node.get_logger().info("Setting OFFBOARD mode...")
    # modes.engage_offboard_mode()

    # Send a few offboard messages before arming (PX4 safety requirement)
    for _ in range(200):
        node.offboard_signal()
        node.pub_att()  # send initial dummy attitude/thrust commands
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("Arming the drone...")
    modes.arm()
    node.armed = True
    rclpy.spin_once(node, timeout_sec=1/30)

    node.get_logger().info("Entering control loop...")

    # Main loop
    i = 0
    PLANNING_BUDGET = 8

    try:
        while rclpy.ok():
            # Check if we are in Offboard mode (14)
            if node.state.nav_state == 14:
                node.pub_att()
                # node.get_action()  
            else:
                pass

            rclpy.spin_once(node, timeout_sec=1.0 / 30.0)
            i += 1

    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
