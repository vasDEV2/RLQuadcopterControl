import rclpy
from rclpy.node import Node
from vehicle import Vehicle
from model import LoadONNX
import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt


def get_input():

    x = input("Target X coordinate: ")
    y = input("Target Y coordinate: ")
    z = input("Target Z coordinate: ")

    return [x, y, z]


class Control(Node):
    
    def __init__(self, target_pos):
        super().__init__('drone_control')   # Node name
        self.get_logger().info("Control has started!")    
        self.time = time.time()    
        self.vehicle = Vehicle(self)
        self.target = target_pos
        self.i = 0
        self.y = 0
   
        self.model = LoadONNX("/home/vasudevan/test_model/test_rigour_1")

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.infinty = self.generate_infinity(num_points=1000)

        # self.infinty = [(5, 5), (4,5), (3,5), (2, 5)]
        self.wp_num = 0
        self.erros = []
        self.errors = 0


    def generate_infinity(self, num_points=1000, scale=1.0):
        """
        Lemniscate of Gerono (∞ shape)
        """
        t = np.linspace(0, 2*np.pi, num_points)
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)
        return np.vstack((x, y)).T


    def generate_s_curve(self, num_points=1000, scale=1.0):
        """
        Smooth S curve using tanh
        """
        t = np.linspace(-2, 2, num_points)
        x = scale * t
        y = scale * np.tanh(t)
        return np.vstack((x, y)).T


    def generate_straight_line(self, num_points=1000, start=(0, 0), end=(5, 0)):
        """
        Straight line interpolation
        """
        x = np.linspace(start[0], end[0], num_points)
        y = np.linspace(start[1], end[1], num_points)
        return np.vstack((x, y)).T
    
    def direct_motor_rebrand(self, wp, height=1.0, T_W=2.11):


        curr = self.vehicle.pos.copy()


        curr[2] = height/5 - curr[2]/5
        curr[0] = wp[0]/5 - curr[0]/5
        curr[1] = wp[1]/5 - curr[1]/5

        self.error = math.sqrt(curr[0]**2 + curr[1]**2)

        q = self.vehicle.quats.copy()

        r = R.from_quat(q)

        ned_to_enu = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
        r_lfu_enu = ned_to_enu @ r.as_matrix() @ ned_to_enu.T

        eu = R.from_matrix(r_lfu_enu).as_euler("ZYX")
        mat_res = r_lfu_enu[0:2, :]
        mat = np.reshape(mat_res, (6))

        obs = np.concatenate([curr, mat, self.vehicle.LV, self.vehicle.AV, [T_W]]).astype(np.float32)
        obs = np.expand_dims(obs, axis=0)

        raw_action = self.model.predict(obs)
        raw_action = raw_action.squeeze(0)
        raw_action = (raw_action+1)/2
        raw_action = np.clip(raw_action, 0.0, 1.0)
        raw_action = raw_action.squeeze(0)
        raw_action = (np.sqrt(raw_action))*1000
        raw_action = (raw_action-150)/850
        raw_action = raw_action.tolist()

        action = [raw_action[2], raw_action[3], raw_action[1], raw_action[0]]

        return action
        

    def timer_callback(self):

        if not self.vehicle.odom:
            return


        if not self.vehicle.offboard:
            self.vehicle.enable_offboard()
            return
        
        if self.y%100 == 0:

            self.kkk += time.time() - self.time
            self.time = time.time()
            print(f"Time elapsed {self.kkk}")

        thrust = self.direct_motor_rebrand(self.infinty[self.wp_num], height=1.0)
        self.y += 1
        if self.y <= 2000:
        # if True:
            self.vehicle.offboard_control("position")
            if self.y <= 2000:
                self.vehicle.set_trajectory([0, 0, -1.0])
            else:
                self.vehicle.set_trajectory([self.infinty[self.wp_num][0], self.infinty[self.wp_num][1], -1.0])
                self.errors = math.sqrt((self.infinty[self.wp_num][0] - self.vehicle.pos[0])**2 + (self.infinty[self.wp_num][1] - self.vehicle.pos[1])**2) 
        else:

            self.vehicle.offboard_control()
            self.vehicle.publish_ctbr(thrust)
            self.errors = math.sqrt((self.infinty[self.wp_num][0] - self.vehicle.pos[0])**2 + (self.infinty[self.wp_num][1] - self.vehicle.pos[1])**2)

        self.erros.append(self.errors)

        if self.errors <= 0.1 and self.errors != 0:
            self.wp_num += 1
        if self.wp_num >= 1000:
            print("MEAN ERROR: ", sum(self.erros)/len(self.erros))
            self.wp_num = 0
        
       

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
