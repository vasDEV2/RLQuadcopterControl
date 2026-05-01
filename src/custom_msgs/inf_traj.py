    import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
from scipy.interpolate import CubicSpline

from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist
from mavros_msgs.msg import Thrust


class EightShapeTrajectory(Node):
    """
    ROS 2 node for generating and publishing a smooth figure-eight trajectory.
    """

    def __init__(self, node_name='eight_shape_trajectory'):
        super().__init__(node_name)
        self.command_active = True

        # Publishers and subscribers
        self.trajectory_pub = self.create_publisher(MultiDOFJointTrajectory, '/command/trajectory', 10)
        self.command_sub = self.create_subscription(
            Thrust,
            '/mavros/setpoint_attitude/thrust',
            self.sp_callback,
            10
        )

    def sp_callback(self, msg):
        """Trigger trajectory start upon receiving a thrust message."""
        if not self.command_active:
            self.get_logger().info("Command received, trajectory will start.")
            self.command_active = True
            self.destroy_subscription(self.command_sub)
            self.command_sub = None

    def generate_8_shape_waypoints(self, a=5.0, b=5.0, height=1.0, num_points=100):
        """
        Generate waypoints for a smooth figure-eight (∞) pattern.
        - a: horizontal radius (x-axis)
        - b: vertical radius (y-axis)
        - height: constant z-height
        - num_points: number of interpolation points
        """
        offset = 1.0
        t = np.linspace(0, 2 * np.pi, num_points)
        ####1
        # x = a * np.sin(t)
        # y = b * np.sin(t) * np.cos(t)
        # z = np.ones_like(x) * height

        ####2
        # y = a * np.sin(t)
        # z = (b * np.sin(t) * np.cos(t)) + height
        # x = np.ones_like(y) * 0

        ####3
        # x = b * np.sin(t) * np.cos(t)
        # y = np.ones_like(x) * 0
        # z = a * np.sin(t) + height

        ####4 line traj in z
        # x = np.zeros_like(t)
        # y = np.zeros_like(t)
        # z = height + a * np.sin(t)

        ####5 line traj in x
        # x = a * np.sin(t)
        # y = np.zeros_like(t)
        # z = height*np.ones_like(t)

        ####6 circle
        y = a * np.cos(t)
        x = a * np.sin(t)
        z = np.ones_like(t) * height

        ####7 helix
        # x = a * np.cos(2*t)
        # y = a * np.sin(2*t)
        # z = (height) + (b) * np.sin(t/2)

        ####8 diagonal
        # s = (np.sin(t) + 1) / 2.0
        # x_start, y_start, z_start = 0.0, 0.0, height
        # x_end,   y_end,   z_end   = a,   a,   height + b
        # x = x_start + s * (x_end - x_start)
        # y = y_start + s * (y_end - y_start)
        # z = z_start + s * (z_end - z_start)

        waypoints = np.stack([x, y, z], axis=1)
        return waypoints  # removed vstack to avoid duplicate

    def generate_square_waypoints(self, side_length=2.0, height=1.0):
        """
        Generates waypoints for a square. The CubicSpline in start_trajectory 
        will naturally smooth the corners for the drone.
        """
        half_s = side_length / 2.0
        
        # Define the 4 corners of the square
        # We repeat the first point at the end to close the loop
        # Format: [x, y, z]
        corners = np.array([
            [ half_s,  half_s, height],
            [-half_s,  half_s, height],
            [-half_s, -half_s, height],
            [ half_s, -half_s, height],
            [ half_s,  half_s, height] 
        ])
        
        return corners

    def start_trajectory(self, waypoints):
        """Generate and publish a smooth trajectory from waypoints."""
        v_max = 0.3
        sampling_hz = 30.0
        dt = 1.0 / sampling_hz

        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        segment_times = distances / v_max
        time_vector = np.concatenate(([0], np.cumsum(segment_times)))

        # Ensure strictly increasing time vector
        if not np.all(np.diff(time_vector) > 0):
            _, unique_idx = np.unique(time_vector, return_index=True)
            time_vector = time_vector[unique_idx]
            waypoints = waypoints[unique_idx]

        total_duration = time_vector[-1]

        cs_x = CubicSpline(time_vector, waypoints[:, 0])
        cs_y = CubicSpline(time_vector, waypoints[:, 1])
        cs_z = CubicSpline(time_vector, waypoints[:, 2])

        t_sample = np.arange(0, total_duration, dt)
        self.get_logger().info(f"Publishing {len(t_sample)} points over {total_duration:.2f} s.")

        for t in t_sample:
            if not rclpy.ok():
                break

            pt = MultiDOFJointTrajectoryPoint()

            transform = Transform()
            transform.translation.x = float(cs_x(t))
            transform.translation.y = float(cs_y(t))
            transform.translation.z = float(cs_z(t))
            transform.rotation.w = 1.0

            velocities = Twist()
            velocities.linear.x = float(cs_x(t, 1))
            velocities.linear.y = float(cs_y(t, 1))
            velocities.linear.z = float(cs_z(t, 1))

            accelerations = Twist()
            accelerations.linear.x = float(cs_x(t, 2))
            accelerations.linear.y = float(cs_y(t, 2))
            accelerations.linear.z = float(cs_z(t, 2))

            pt.transforms.append(transform)
            pt.velocities.append(velocities)
            pt.accelerations.append(accelerations)

            traj_msg = MultiDOFJointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.header.frame_id = 'map'
            traj_msg.points.append(pt)

            self.trajectory_pub.publish(traj_msg)
            time.sleep(dt)

        self.get_logger().info("Completed one 8-shape cycle.")


def main(args=None):
    rclpy.init(args=args)
    node = EightShapeTrajectory()

    node.get_logger().info("Waiting for subscriber on /command/trajectory...")
    while node.trajectory_pub.get_subscription_count() == 0 and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1.0)

    node.get_logger().info("Subscriber connected. Waiting for command on /mavros/setpoint_attitude/thrust...")

    while rclpy.ok():
        if node.command_active:
            # Trajectory parameters
            height = 1.0
            a, b = 0.6, 0.6  # shape size 3.0, 2.7   #train: 2.7, 3.0
            waypoints = node.generate_8_shape_waypoints(a, b, height)
            # waypoints = node.generate_square_waypoints(a, height)

            # 🔁 Continuously run trajectory in loop
            while rclpy.ok():
                start_time = node.get_clock().now()
                node.start_trajectory(waypoints)
                duration = (node.get_clock().now() - start_time).nanoseconds / 1e9
                node.get_logger().info(f"One trajectory cycle took {duration:.2f} s.")
                # optional: small pause between cycles
                time.sleep(1.0)

        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("Script finished.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

