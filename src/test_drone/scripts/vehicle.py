import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry, SensorCombined, VehicleRatesSetpoint, TrajectorySetpoint, VehicleCommand, VehicleStatus, OffboardControlMode, VehicleAttitude, VehicleTorqueSetpoint, ActuatorMotors
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import torch
from enum import IntEnum
import numpy as np

class PX4FlightMode(IntEnum):
    MANUAL = 1
    ALTCTL = 2          # Altitude control
    POSCTL = 3          # Position control
    AUTO_MISSION = 4
    AUTO_LOITER = 5
    OFFBOARD = 6
    STABILIZED = 7
    RATTITUDE = 8
    ACRO = 9
    AUTO_TAKEOFF = 10
    AUTO_LAND = 11
    AUTO_FOLLOW_TARGET = 12  # (if supported)
    AUTO_PRECLAND = 13       # Precision land
    ORBIT = 14             

class Vehicle():

    def __init__(self, node):
        
        self.node = node
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.pos = None
        self.LA = None
        self.AV = None
        self.arming_state = None
        self.nav_state = None
        self.offboard = False
        self.sp_count = 0
        self.quats = [1,2,1,2]

        # ---- Examples: Create a subscription ----
        self.odometry_sub = self.node.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odometry_callback, qos_sub)
        self.imu_gyro_sub = self.node.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.imu_gyro_callback, qos_sub)
        self.status_sub = self.node.create_subscription(VehicleStatus, 'fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_sub)
        self.attitude_sub = self.node.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_sub)
        

        self.ctbr_pub = self.node.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_pub)
        self.torque_pub = self.node.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_pub)
        self.motor_pub = self.node.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_pub)
        self.publisher_trajectory = self.node.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_pub)
        self.cmd_pub = self.node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_pub)
        self.publisher_offboard_mode = self.node.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_pub)

    def odometry_callback(self, msg):

        self.pos = [-msg.position[0]/10, msg.position[1]/10, -msg.position[2]/10]
        self.LV = [-msg.velocity[0]/10, msg.velocity[1]/10, -msg.velocity[2]/10]
        # self.LV = msg.velocity/10
        # self.LV[2] = -self.LV[2]

    def attitude_callback(self, msg):

        quats = msg.q

        self.quats[0] = quats[1]
        self.quats[1] = quats[2]
        self.quats[2] = quats[3]
        self.quats[3] = quats[0]

    def imu_gyro_callback(self, msg):
        self.LA = msg.accelerometer_m_s2
        self.AV = [msg.gyro_rad[1], msg.gyro_rad[0], -msg.gyro_rad[2]]
        # self.AV = msg.gyro_rad

    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def timer_callback(self):
        self.node.get_logger().info("Timer callback triggered")

    def change_mode(self, mode):
        msg = VehicleCommand()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)

        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0           # base mode flag (custom mode)
        msg.param2 = float(mode)           # OFFBOARD mode number
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.cmd_pub.publish(msg)
        self.node.get_logger().info("Sent OFFBOARD mode command")

    def arm_vehicle(self):

        msg = VehicleCommand()
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)

        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0     # ARM
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.cmd_pub.publish(msg)
        # self.node.get_logger().info("Sent ARM command")

    def offboard_control(self):

        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)

        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        offboard_msg.thrust_and_torque = False
        offboard_msg.direct_actuator = False

        self.publisher_offboard_mode.publish(offboard_msg)

    def enable_offboard(self):

        self.offboard_control()
        if self.sp_count == 20:

            self.set_trajectory([0, 0, 0])
            self.arm_vehicle()
            self.node.get_logger().info('Vehicle ARMED!')
            self.offboard = True
        
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.change_mode(PX4FlightMode.OFFBOARD)
            self.set_trajectory([0, 0, 0])

        self.set_trajectory([0,0,0])

        if self.sp_count <= 21:
            self.sp_count += 1

    def set_trajectory(self, pos):

        trajectory_msg = TrajectorySetpoint()

    
        trajectory_msg.position[0] = pos[0]
        trajectory_msg.position[1] = pos[1]
        trajectory_msg.position[2] = pos[2]
        self.publisher_trajectory.publish(trajectory_msg)

    def publish_ctbr(self, rates):

        msg = VehicleRatesSetpoint()
        msg2 = VehicleTorqueSetpoint()
        msg3 = ActuatorMotors()
        print(rates)
        # msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        # msg.roll = rates[1].item()
        # msg.pitch = rates[2].item()
        # msg.yaw = 0.
        # if -rates[0] < -1:
        #     rates[0] = 1
        # elif -rates[0] > 0:
        #     rates[0] = 0
        msg.thrust_body = [0., 0., -rates[0]]
        msg.roll = rates[2]
        msg.pitch = rates[1]
        msg.yaw = rates[3]
        # msg2.xyz = [0., 0., 1.0]
        # rates = rates.squeeze(0)
        # rates = rates.squeeze(0)
        # rates = rates.numpy()
        # msg3.NUM_CONTROLS = 4
        # msg3.control = [rates[0], rates[2], rates[3], rates[1], 0., 0., 0.,0.,0.,0.,0.,0.]
        # msg3.control = [1., 1., 0., 0., 0., 0., 0.,0.,0.,0.,0.,0.]
        # msg3.control = [1., 0., 1., 0.]
        # msg3.control = [-rates[0], -rates[1], , 0.99937262, 0., 0., 0.,0.,0.,0.,0.,0.]

        print(f"MESSAGE: {msg}")

        self.ctbr_pub.publish(msg)
