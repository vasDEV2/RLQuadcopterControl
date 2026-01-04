import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry, SensorCombined, VehicleRatesSetpoint, TrajectorySetpoint, VehicleCommand, VehicleStatus, OffboardControlMode
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
# import torch
from vehicle import Vehicle
from model import LoadONNX
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

import numpy as np
from scipy.spatial.transform import Rotation as R

def gravity_in_frd(px4_quat, vector):
    """
    Convert gravity vector into PX4 FRD body frame, given PX4 quaternion.   

    PX4 quaternion = rotation from BODY(FRD) → NED.

    Args:
        px4_quat: [w, x, y, z] quaternion (BODY → NED)
        g: gravity magnitude

    Returns:
        np.array([gx, gy, gz]) in FRD frame
    """

    # PX4 gives quaternion as [w, x, y, z]   (scalar-first)
    w, x, y, z = px4_quat

    # SciPy expects [x, y, z, w] (vector-first)
    scipy_quat = np.array([x, y, z, w])

    # Create BODY→NED rotation
    R_body_to_ned = R.from_quat(scipy_quat)

    # To express a NED vector in BODY frame, invert the rotation
    R_ned_to_body = R_body_to_ned.inv()

    # Gravity in NED (+Z downward)
    g_ned = vector

    # Transform gravity into FRD frame
    g_frd = R_ned_to_body.apply(g_ned)

    return g_frd


def get_input():

    x = input("Target X coordinate: ")
    y = input("Target Y coordinate: ")
    z = input("Target Z coordinate: ")

    return [x, y, z]


class Control(Node):
    
    def __init__(self, target_pos):
        super().__init__('drone_control')   # Node name
        self.get_logger().info("Control has started!")        
        self.vehicle = Vehicle(self)
        self.target = target_pos
        self.i = 0
        self.yaw_rate = 0
        self.sim_dt = 0.02
        # self.model = ModelLoader("policy.pt")
        self.model = LoadONNX("models/EpisodeV.onnx")

        self.Gz = -9.81
        self.mass = 2
        self.inertia = np.array([[0.02166666666666667, 0, 0], 
                                [0, 0.02166666666666667, 0],
                                [0, 0, 0.04000000000000001]])
        
        self.kin_inv = np.array([[16.6, 0, 0], 
                                [0, 16.6, 0],
                                [0, 0, 5.0]])
        
        arm_l = 0.17
        kappa = 0.016
        
        t_BM = (arm_l * np.sqrt(0.5) *
                    np.array([
                        [ 1, -1, -1,  1],
                        [-1, -1,  1,  1],
                        [ 0,  0,  0,  0]
                    ])
                )
        
        self.allocation_matrix = np.vstack([ np.ones(4),
                                        t_BM[:2, :],
                                        kappa * np.array([1, -1, 1, -1])
                                    ])
        
        self.inverse_allocat = np.linalg.inv(self.allocation_matrix)
    
        # print(f'INERTIA: {self.inertia}')

        self.mean = np.ones(4)*(-self.Gz)
        self.std = np.ones(4)*(-self.Gz)*0.35

        self.total_thrust = self.mean*1.35

        self.i = 0

        self.action_array = np.load("/home/vasudevan/Desktop/flightmare/flightrl/examples/actions_anh.npy")

        # ---- Examples: Create a timer ----
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.a = []
        self.maxthrst = 9.4618

    def get_thrust(self, action):

        action[0] = action[0]*self.mass

        # err = action[1:4] - self.vehicle.AV
        # print(f"error {err}")
        # print(f"av {self.vehicle.AV}")

        # action[1:4] = self.inertia @ self.kin_inv @ err + np.cross(self.vehicle.AV, self.inertia @ self.vehicle.AV)

        # print(f"ACTION 1: {action}")
        # action[1:4] = [0, 0, 0]

        # action[3] = 0.0

        thrust = action
        print(thrust.shape)

        # thrust = self.inverse_allocat @ action

        # print(f"ACTION 2: {thrust}")

        # thrust /= (9.81*2*self.mass/4)
        # thrust /= self.maxthrst

        thrust[0] = thrust[0]/(self.maxthrst*4)

        thrust[0] = np.clip(thrust[0], 0, 1.0)
        # print(f"ACTION 3: {thrust}")
        # omega_cmd = np.sqrt(thrust)
        thrust[0] = np.sqrt(thrust[0])

        # print(f"ACTION 4: {omega_cmd}")


        return thrust

    def get_control(self):

        print(f"CURRENT POS: {self.vehicle.pos}")
        print(f"CURRENT VEL: {self.vehicle.LV}")
        curr = self.vehicle.pos.copy()
        # curr[2] = -curr[2]
        # temp = curr[0]
        # curr[0] = curr[1]
        # curr[1] = temp
        # curr = curr/10
        # cv = self.vehicle.LV.copy()
        # cv[2] = -cv[2]
        # temp = cv[0]
        # cv[0] = cv[1]
        # cv[1] = temp
        # curr[0] = curr[0]
        curr[0] = curr[0] - 0.5
        # curr[1] = curr[1] + 0.5
        # curr[2] = curr[2] + (0.5 - 0.05)
        # print("goal_local: ", goal_local)
        # eu = [0, 0, 0]
        # r = R.from_euler("zyx", eu)
        r = R.from_quat(self.vehicle.quats)
        print(self.vehicle.quats)

        ned_to_enu = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
        r_lfu_enu = ned_to_enu @ r.as_matrix() @ ned_to_enu.T

        print(f"quats: {R.from_matrix(r_lfu_enu).as_quat()}")
        # print(r.as_matrix())
        
        # r_enu = ned_to_enu @ r.as_matrix()

        # print(r_enu)

        angst = R.from_matrix(r_lfu_enu).as_euler("ZYX", degrees=True)

        # angst[0] = 90 + angst[0]
        angst[0] = 0

        rrk = R.from_euler("ZYX", angst, degrees=True).as_matrix()

        # ang[0] = 0
        # ang[0] = 90-ang[0]
        # ang[2] = -ang[2]
        # ang[2] = -ang[2]
        # print(f"EULER ZYX: {ang}")
        rr = R.from_euler("ZXY", angst, degrees=True)


        # temp = ang[2]
        # ang[2] = ang[1]
        # ang[1] = temp
        # rr = R.from_euler
        print(f"ANGLE : {angst}")
        # euler = rr.as_matrix()
        # euler = r.as_matrix()

        # euler = np.reshape(euler, (9,), "F")
        # euler = np.reshape(euler, (9,), "F")
        euler = np.reshape(rrk, (9,), "F")
        # print(len(self.vehicle.pos))
        obs = np.concatenate([curr, euler, self.vehicle.LV, self.vehicle.AV, [self.yaw_rate]]).astype(np.float32)

        obs = np.expand_dims(obs, axis=0)
        # print(f"OBSERVATION: {obs}")
        raw_action = self.model.predict(obs)
        print(f"RAW_ACTION: {raw_action}")

        # raw_action = raw_action.numpy()
        print(raw_action.shape)

        action = np.zeros((raw_action.shape))

        # # action[:, :, 0] = self.std[0]*raw_action[:, :, 0] + self.mean[0]
        action[:, :, 0] = raw_action[:, :, 0] + 9.81
        # action[0] = action[0] + 9.81
        # # if action[:, :, 0] > self.total_thrust[0]:
        #     # action[:, :, 0] = self.total_thrust[0]

        # # action[:, :, 0] = raw_action[:, :, 0]*self.mean[0]
        action[:, :, 1:3] = raw_action[:, :, 1  :3]*5
        # print(f"RAW_ACTION 2: {action}")
        # action[:,:, 1] = action[:,:, 1]
        # action[:, :, 3] = raw_action[:, :, 3]*5
        # action[:, :, 1] = 0
        # action[1:4] = action[1:4]*5
        action = action.squeeze(0)
        action = action.squeeze(0)
        # action[3] = -raw_action[:,:, 3]
        print(action.shape)
        # action[0] = min(action[0], 9.81*1.39)
        # action[0] = action[0]/(9.81*1.39)
       
        thrust = self.get_thrust(action)

        print(f"Final trhsut {thrust}")

        self.yaw_rate += 0*self.sim_dt

        return thrust
    
    def get_flightmare_policy(self, raw_action):

        # print(self.vehicle.pos)
        # print(self.target)
        # goal_local = self.target - self.vehicle.pos

        # r = R.from_quat(self.vehicle.quats)
        # euler = r.as_matrix()

        # euler = np.reshape(euler, (9,))

        # obs = np.concatenate([goal_local, euler, self.vehicle.LV, self.vehicle.AV])
        # # obs[3] =  0.0
        # obs = obs.astype(np.float32)
        # obs = np.expand_dims(obs, axis=0)
        # print(f"observation: {obs}")
        # # obs = np.array([[ 0.20710441,  0.17971317,  0.17334747,  0.31125432,  0.01186327,  0.01035172,
        #     # -0.10929969, -0.04562206,  0.02839059, -0.04129951,  0.01239144, -0.022663  ]]).astype(np.float32)
        # raw_action = self.model.predict(obs)
            
        print(f"RAW ACTION {raw_action}")

        action_processed = raw_action*self.std + self.mean

        print(f"PROCCESSED : {action_processed}")

        print("STD: ", self.std)

        normalized_thrust = action_processed / self.total_thrust

        print(f"normalized thrust : {normalized_thrust}")

        return normalized_thrust 

    def timer_callback(self):

        if not self.vehicle.offboard:
            self.vehicle.enable_offboard()
            return
        
        # yo = np.array([ 18.8423, 0.00409635, 0.010741, 0.2])

        # yoyo = self.inverse_allocat @ yo

        # print(yoyo)
        
        # if self.i < 300:
        #     # print(f"allo: {self.allocation_matrix}")
        #     # print(f"allo I: {self.inverse_allocat}")
        #     print(self.action_array[self.i, :])
        #     thrust = self.get_control(self.action_array[self.i, :])
        #     self.vehicle.offboard_control()
        #     self.vehicle.publish_ctbr(thrust)
        #     self.i += 1
        thrust = self.get_control()
        # thrust = [1,2,3,4]
        self.vehicle.offboard_control()
        # print(f"THRUST : {thrust}")
        self.vehicle.publish_ctbr(thrust)

        # print("ACC: ", self.vehicle.LA[2])

        # self.vehicle.set_trajectory([5, 0, -5])
        
       

def main(args=None):
    rclpy.init(args=args)
    # target = get_input()
    target = [0.5,0,0]
    node = Control(target)
    rclpy.spin(node)     # Keep node alive
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
