import rclpy
from rclpy.node import Node
import numpy as np
import time
from scipy.interpolate import CubicSpline
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist


class SmoothRandomTrajectory(Node):
    """
    ROS 2 node that continuously generates smooth random 3D waypoints
    and interpolates a trajectory between them with varying velocities.
    """

    def __init__(self):
        super().__init__('smooth_random_trajectory')

        self.trajectory_pub = self.create_publisher(MultiDOFJointTrajectory, '/command/trajectory', 10)

        # Bounds for position and velocity
        self.bounds = {'x': (-2.0, 2.0), 'y': (-2.0, 2.0), 'z': (2.5, 5.0)}
        self.vel_range = (0.3, 0.8)

        self.prev_point = np.array([0.0, 0.0, 1.0])  # Start position

        self.timer = self.create_timer(0.1, self.publish_next_trajectory)  # 10Hz loop

        self.get_logger().info("Smooth random trajectory node started.")

    def random_point(self):
        """Generate a random target point within bounds."""
        return np.array([
            np.random.uniform(*self.bounds['x']),
            np.random.uniform(*self.bounds['y']),
            np.random.uniform(*self.bounds['z'])
        ])

    def publish_next_trajectory(self):
        """Generate and publish a short smooth trajectory to a new random point."""
        new_point = self.random_point()
        v_max = np.random.uniform(*self.vel_range)

        # Generate smooth spline between prev and new point
        num_samples = 100
        distance = np.linalg.norm(new_point - self.prev_point)
        total_time = max(7.0, distance / v_max)  # at least 2 seconds per move
        t = np.linspace(0, total_time, num_samples)

        waypoints = np.vstack([
            np.linspace(self.prev_point[0], new_point[0], num_samples),
            np.linspace(self.prev_point[1], new_point[1], num_samples),
            np.linspace(self.prev_point[2], new_point[2], num_samples)
        ]).T

        # Smooth with cubic splines
        cs_x = CubicSpline(t, waypoints[:, 0])
        cs_y = CubicSpline(t, waypoints[:, 1])
        cs_z = CubicSpline(t, waypoints[:, 2])

        dt = total_time / num_samples
        for i in range(num_samples):
            if not rclpy.ok():
                break

            ti = t[i]
            pt = MultiDOFJointTrajectoryPoint()

            transform = Transform()
            transform.translation.x = float(cs_x(ti))
            transform.translation.y = float(cs_y(ti))
            transform.translation.z = float(cs_z(ti))
            transform.rotation.w = 1.0

            velocities = Twist()
            velocities.linear.x = float(cs_x(ti, 1))
            velocities.linear.y = float(cs_y(ti, 1))
            velocities.linear.z = float(cs_z(ti, 1))

            accelerations = Twist()
            accelerations.linear.x = float(cs_x(ti, 2))
            accelerations.linear.y = float(cs_y(ti, 2))
            accelerations.linear.z = float(cs_z(ti, 2))

            pt.transforms.append(transform)
            pt.velocities.append(velocities)
            pt.accelerations.append(accelerations)

            traj_msg = MultiDOFJointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.header.frame_id = 'map'
            traj_msg.points.append(pt)

            self.trajectory_pub.publish(traj_msg)
            time.sleep(dt)

        self.prev_point = new_point  # Update start point
        self.get_logger().info(f"Moved smoothly to {new_point} at v={v_max:.2f} m/s.")


def main(args=None):
    rclpy.init(args=args)
    node = SmoothRandomTrajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
