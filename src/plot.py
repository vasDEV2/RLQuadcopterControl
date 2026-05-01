# import rclpy
# from rclpy.node import Node
# from px4_msgs.msg import VehicleOdometry
# from rclpy.qos import qos_profile_sensor_data
# import matplotlib.pyplot as plt
# import time


# class TrajectoryPlotter(Node):

#     def __init__(self):
#         super().__init__('trajectory_plotter')

#         self.x_data = []
#         self.y_data = []

#         self.last_msg_time = time.time()

#         self.subscription = self.create_subscription(
#             VehicleOdometry,
#             '/fmu/out/vehicle_odometry',
#             self.callback,
#             qos_profile_sensor_data
#         )

#         # check every second if bag stopped
#         self.timer = self.create_timer(1.0, self.check_timeout)

#         self.get_logger().info("Waiting for odometry data...")

#     def callback(self, msg):

#         x = msg.position[0]
#         y = msg.position[1]

#         self.x_data.append(x)
#         self.y_data.append(y)

#         self.last_msg_time = time.time()

#         # print(f"X: ")

#     def check_timeout(self):

#         # if no message for 3 seconds → bag finished
#         if time.time() - self.last_msg_time > 3:

#             self.get_logger().info("Bag finished. Plotting trajectory.")

#             plt.plot(self.x_data, self.y_data)
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.title("Vehicle trajectory")
#             plt.axis("equal")
#             plt.grid(True)
#             plt.show()

#             rclpy.shutdown()


# def main():

#     rclpy.init()
#     node = TrajectoryPlotter()

#     rclpy.spin(node)


# if __name__ == "__main__":
#     main()

# import rclpy
# from rclpy.node import Node
# from px4_msgs.msg import VehicleOdometry
# from trajectory_msgs.msg import MultiDOFJointTrajectory
# from rclpy.qos import qos_profile_sensor_data

# import matplotlib.pyplot as plt
# import time


# class TrajectoryComparison(Node):

#     def __init__(self):
#         super().__init__('trajectory_comparison')

#         # storage
#         self.drone_x = []
#         self.drone_y = []

#         self.cmd_x = []
#         self.cmd_y = []

#         self.last_msg_time = time.time()

#         # drone trajectory
#         self.create_subscription(
#             VehicleOdometry,
#             '/fmu/out/vehicle_odometry',
#             self.odom_callback,
#             qos_profile_sensor_data
#         )

#         # commanded trajectory
#         self.create_subscription(
#             MultiDOFJointTrajectory,
#             '/command/trajectory',
#             self.command_callback,
#             10
#         )

#         self.get_logger().info("Recording trajectories...")
#         self.timer = self.create_timer(1.0, self.check_timeout)

#     def odom_callback(self, msg):

#         x = msg.position[0]
#         y = msg.position[1]

#         self.drone_x.append(x)
#         self.drone_y.append(-y)

#     def command_callback(self, msg):

#         if len(msg.points) == 0:
#             return

#         transform = msg.points[0].transforms[0]

#         x = transform.translation.x
#         y = transform.translation.y

#         self.cmd_x.append(x)
#         self.cmd_y.append(y)
#         self.last_msg_time = time.time()
#         # print("heyyy")

#     def check_timeout(self):

#         # if no message for 3 seconds → bag finished
#         if time.time() - self.last_msg_time > 3:

#             # Plot trajectories
#             plt.figure()

#             plt.plot(self.drone_x, self.drone_y, label="Drone trajectory", linewidth=2)
#             plt.plot(self.cmd_x, self.cmd_y, '--', label="Commanded trajectory", linewidth=2)

#             plt.xlabel("X (m)")
#             plt.ylabel("Y (m)")
#             plt.title("Trajectory Tracking")    

#             plt.legend()
#             plt.axis("equal")
#             plt.grid(True)

#             plt.show()


# def main():

#     rclpy.init()

#     node = TrajectoryComparison()

#     try:
#         rclpy.spin(node)

#     except KeyboardInterrupt:
#         pass

#     node.destroy_node()
#     rclpy.shutdown()



# if __name__ == "__main__":
#     main()

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from mpl_toolkits.mplot3d import Axes3D
from rclpy.qos import qos_profile_sensor_data

import matplotlib.pyplot as plt
import time
import numpy as np




class LiveTrajectoryPlot(Node):

    def __init__(self):
        super().__init__('trajectory_plot')

        self.drone_x = []
        self.drone_y = []
        self.drone_z = []

        self.cmd_x = []
        self.cmd_y = []
        self.cmd_z = []

        self.last_msg_time = time.time()

        # subscriptions
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            MultiDOFJointTrajectory,
            '/command/trajectory',
            self.cmd_callback,
            10
        )

        self.comapre = False

        # setup plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig_e, self.ax_e = plt.subplots(3)

        self.drone_line, = self.ax.plot([], [], 'b', label="Drone trajectory")
        self.cmd_line, = self.ax.plot([], [], 'r--', label="Commanded trajectory")
        self.x_error, = self.ax_e[0].plot([], 'r', label="X")
        self.y_error, = self.ax_e[1].plot([], 'b', label="Y")
        self.z_error, = self.ax_e[2].plot([], 'g', label="Z")


        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Trajectory Tracking")
        self.ax.legend()
        # self.ax.grid(True)

        self.ax_e[0].set_ylabel("X error (m)")
        self.ax_e[1].set_ylabel("Y error (m)")
        self.ax_e[2].set_ylabel("Z error (m)")
        self.ax_e[2].set_xlabel("Time")
        # self.ax_e.legend()

        # update plot timer
        self.timer = self.create_timer(0.1, self.update_plot)

    # def odom_callback(self, msg):

    #     if self.comapre:

    #         self.drone_x.append(msg.position[0])
    #         self.drone_y.append(-msg.position[1])
    #         self.drone_z.append(msg.position[2])

    #     self.last_msg_time = time.time()

    def odom_callback(self, msg):

        if self.comapre and hasattr(self, "cmd_buffer") and len(self.cmd_buffer) > 0:

            odom_time = msg.timestamp * 1e-6   # PX4 timestamp is in microseconds

            # find command with closest timestamp
            closest_cmd = min(
                self.cmd_buffer,
                key=lambda c: abs(c[0] - odom_time)
            )

            self.drone_x.append(msg.position[0])
            self.drone_y.append(-msg.position[1])
            self.drone_z.append(-msg.position[2])

            self.cmd_x.append(closest_cmd[1])
            self.cmd_y.append(closest_cmd[2])
            self.cmd_z.append(closest_cmd[3])

            # print("actual:", msg.position[2])

        self.last_msg_time = time.time()

    # def cmd_callback(self, msg):

    #     if len(msg.points) == 0:
    #         return

    #     t = msg.points[0].transforms[0]

    #     self.cmd_x.append(t.translation.x)
    #     self.cmd_y.append(t.translation.y)
    #     self.cmd_z.append(t.translation.z)

    #     self.comapre = True
    def cmd_callback(self, msg):

        if len(msg.points) == 0:
            return

        t = msg.points[0].transforms[0]

        # store command with timestamp
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not hasattr(self, "cmd_buffer"):
            self.cmd_buffer = []

        self.cmd_buffer.append((
            timestamp,
            t.translation.x,
            t.translation.y,
            t.translation.z
        ))

        # print("command", t.t  ranslation.z)

        self.comapre = True

        

    def update_plot(self):

  
        n = min(len(self.cmd_x), len(self.drone_x))

        print(len(self.cmd_x))
        print(len(self.drone_x))

        self.errorx = np.array(self.cmd_x[:n]) - np.array(self.drone_x[:n])
        self.errory = np.array(self.cmd_y[:n]) - np.array(self.drone_y[:n])
        self.errorz = np.array(self.cmd_z[:n]) - np.array(self.drone_z[:n])
        t = np.arange(len(self.errorx))

        # update live plot
        self.drone_line.set_data(self.drone_x, self.drone_y)
        self.cmd_line.set_data(self.cmd_x, self.cmd_y)

        self.x_error.set_data(t, self.errorx)
        self.y_error.set_data(t, self.errory)
        self.z_error.set_data(t, self.errorz)

        self.ax.relim()
        self.ax_e[0].relim()
        self.ax_e[1].relim()
        self.ax_e[2].relim()
        self.ax.autoscale_view()
        self.ax_e[0].autoscale_view()
        self.ax_e[1].autoscale_view()
        self.ax_e[2].autoscale_view()

        self.fig.canvas.draw()
        self.fig_e.canvas.draw()
        # self.fig_e[1].canvas.draw()
        # self.fig_e[2].canvas.draw()
        self.fig.canvas.flush_events()
        self.fig_e.canvas.flush_events()
        # self.fig_e[1].canvas.flush_events()
        # self.fig_e[2].canvas.flush_events()

        # detect bag finished
        # if time.time() - self.last_msg_time > 3:

        #     self.get_logger().info("Bag finished — showing final plot")

        #     plt.ioff()
        #     plt.figure()

        #     plt.plot(self.drone_x, self.drone_y, label="Drone trajectory")
        #     plt.plot(self.cmd_x, self.cmd_y, '--', label="Commanded trajectory")

        #     figs, axes = plt.subplots(3)
        #     t = np.arange(len(self.errorx))
        #     axes[0].plot(t[:100], self.errorx[:100], 'r', label="X")
        #     axes[1].plot(t[:100], self.errory[:100], 'b', label="Y")
        #     axes[2].plot(t[:100], self.errorz[:100], 'g', label="Z")

        #     axes[0].set_xlabel("X error (m)")
        #     axes[1].set_xlabel("Y error (m)")
        #     axes[2].set_xlabel("Z error (m)")

        #     plt.xlabel("X (m)")
        #     plt.ylabel("Y (m)")
        #     plt.title("Final Trajectory")
        #     plt.legend()
        #     plt.axis("equal")
        #     # plt.grid(True)

        #     plt.show()

        #     rclpy.shutdown()
        if time.time() - self.last_msg_time > 3:

            self.get_logger().info("Bag finished — showing final plot")

            plt.ioff()

            # ---- Trajectory plot ----
            fig_traj = plt.figure(figsize=(16, 12))

            # plt.plot(self.drone_x[:100], self.drone_y[:100], label="Drone trajectory")
            # plt.plot(self.cmd_x[:100], self.cmd_y[:100], '--', label="Commanded trajectory")

            # plt.xlabel("X (m)")
            # plt.ylabel("Y (m)")
            # plt.title("Final Trajectory")
            # plt.legend()
            # plt.axis("equal")
            ax = fig_traj.add_subplot(111, projection='3d')

            # plot trajectories
            ax.plot(self.drone_x[:120], self.drone_y[:120], self.drone_z[:120], label="Drone trajectory", linewidth=2)
            ax.plot(self.cmd_x[:120], self.cmd_y[:120], self.cmd_z[:120], '--', label="Commanded trajectory", linewidth=2)

            # labels
            ax.set_xlabel("X (m)", fontweight='bold')
            ax.set_ylabel("Y (m)", fontweight='bold')
            ax.set_zlabel("Z (m)", fontweight='bold')
            # ax.set_title("Trajectory")

            ax.set_zlim(-5, 5)   # increase vertical range
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # equal scaling
            # ax.set_box_aspect([1,1,1])

            # ax.legend()

            # plt.show()


            fig_traj.savefig("/home/vasudevan/Desktop/trajectory_plot.png", dpi=300)

            # ---- Error plots ----
            fig_err, axes = plt.subplots(3, 1, sharex=True)

            t = np.arange(len(self.errorx))

            axes[0].plot(t[:120], self.errorx[:120], 'r')
            axes[1].plot(t[:120], self.errory[:120], 'b')
            axes[2].plot(t[:120], self.errorz[:120], 'g')

            axes[0].set_ylabel("X error (m)", fontweight='bold')
            axes[1].set_ylabel("Y error (m)", fontweight='bold')
            axes[2].set_ylabel("Z error (m)", fontweight='bold')

            # axes[2].set_xlabel("Time step")

            for ax in axes:
            #     ax.legend()
                ax.grid(True)

            fig_err.savefig("/home/vasudevan/Desktop/error.png", dpi=300)

            plt.show()

            print(f"X RMSE: {np.sqrt(np.mean(np.square(self.errorx)))}")
            print(f"Y RMSE: {np.sqrt(np.mean(np.square(self.errory)))}")
            print(f"Z RMSE: {np.sqrt(np.mean(np.square(self.errorz)))}")

            rclpy.shutdown()


def main():

    rclpy.init()

    node = LiveTrajectoryPlot()

    rclpy.spin(node)


if __name__ == "__main__":
    main()