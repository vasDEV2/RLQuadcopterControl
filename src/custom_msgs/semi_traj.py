import rclpy
from rclpy.node import Node
import numpy as np
import time
from scipy.interpolate import CubicSpline

from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist, Quaternion
from mavros_msgs.msg import Thrust


class SemicircleTrajectory(Node):
    """
    Trajectory (top-down view):
      START → → (pure straight) → → ⌒ semicircle ⌒ → END
    
    Orientation:
      The drone continuously yaws to face the CENTER of the semicircle.
    """

    def __init__(self, node_name='semicircle_trajectory'):
        super().__init__(node_name)
        self.command_active = True

        self.trajectory_pub = self.create_publisher(
            MultiDOFJointTrajectory, '/command/trajectory', 10
        )
        self.command_sub = self.create_subscription(
            Thrust,
            '/mavros/setpoint_attitude/thrust',
            self.sp_callback,
            10
        )

    def sp_callback(self, msg):
        if not self.command_active:
            self.get_logger().info("Command received, trajectory will start.")
            self.command_active = True
            self.destroy_subscription(self.command_sub)
            self.command_sub = None

    def get_quaternion_from_yaw(self, yaw):
        """
        Converts a yaw angle (in radians) to a geometry_msgs Quaternion.
        Assumes roll and pitch are zero.
        """
        # q = [x, y, z, w]
        # w = cos(yaw/2), z = sin(yaw/2), x=0, y=0
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = float(np.sin(yaw / 2.0))
        q.w = float(np.cos(yaw / 2.0))
        return q

    def generate_trajectory(self,
                            start, end,
                            height=1.0,
                            straight_ratio=0.25,
                            num_points=300):
        """
        Returns waypoints including computed Yaw to face the center.
        Output shape: (num_points, 4) -> [x, y, z, yaw]
        """
        x0, y0 = start
        x1, y1 = end

        # Vector pointing from start to end
        fwd = np.array([x1 - x0, y1 - y0], dtype=float)
        total_dist = np.linalg.norm(fwd)
        
        if total_dist > 0:
            fwd /= total_dist
            
        # Normal vector pointing Left
        left_normal = np.array([-fwd[1], fwd[0]])

        # Geometric Parameters
        d_straight = total_dist * straight_ratio
        radius = (total_dist - d_straight) / 2.0

        # Calculate the Coordinates of the Semicircle Center
        # The center is located along the forward axis, at distance (d_straight + radius)
        center_pos = np.array([x0, y0]) + fwd * (d_straight + radius)

        t_vals = np.linspace(0.0, 1.0, num_points)
        xs = np.zeros(num_points)
        ys = np.zeros(num_points)
        yaws = np.zeros(num_points)

        for i, t in enumerate(t_vals):
            if t <= straight_ratio:
                # ── Straight Section ────────────────────────
                local_x = total_dist * t
                local_y = 0.0
            else:
                # ── Semicircle Section ──────────────────────
                t_arc = (t - straight_ratio) / (1.0 - straight_ratio)
                theta = t_arc * np.pi

                # Semicircle geometry
                local_x = d_straight + radius - radius * np.cos(theta)
                local_y = radius * np.sin(theta)

            # Map to global position
            pos = np.array([x0, y0]) + fwd * local_x + left_normal * local_y
            xs[i] = pos[0]
            ys[i] = pos[1]

            # ── Calculate Yaw ──────────────────────────────
            # Vector from Current Drone Position -> Center Position
            vec_to_center = center_pos - pos
            # Compute angle
            yaws[i] = np.arctan2(vec_to_center[1], vec_to_center[0])

        zs = np.ones(num_points) * height

        # Stack x, y, z, and yaw
        return np.stack([xs, ys, zs, yaws], axis=1)

    def start_trajectory(self, waypoints):
        v_max = 0.2
        sampling_hz = 30.0
        dt = 1.0 / sampling_hz

        # Extract position (first 3 cols) for timing calculation
        pos_pts = waypoints[:, :3]
        
        distances = np.linalg.norm(np.diff(pos_pts, axis=0), axis=1)
        segment_times = distances / v_max
        time_vector = np.concatenate(([0], np.cumsum(segment_times)))

        if not np.all(np.diff(time_vector) > 0):
            _, unique_idx = np.unique(time_vector, return_index=True)
            time_vector = time_vector[unique_idx]
            waypoints = waypoints[unique_idx]

        total_duration = time_vector[-1]

        # Splines for Position
        cs_x = CubicSpline(time_vector, waypoints[:, 0])
        cs_y = CubicSpline(time_vector, waypoints[:, 1])
        cs_z = CubicSpline(time_vector, waypoints[:, 2])
        
        # Spline for Yaw (orientation)
        # Note: In this specific geometry, yaw varies continuously (approx 0 -> -pi),
        # so we don't need complex unwrapping logic.
        cs_yaw = CubicSpline(time_vector, waypoints[:, 3])

        t_sample = np.arange(0, total_duration, dt)
        self.get_logger().info(
            f"Publishing {len(t_sample)} points over {total_duration:.2f} s."
        )

        for t in t_sample:
            if not rclpy.ok():
                break

            pt = MultiDOFJointTrajectoryPoint()
            transform = Transform()

            # 1. Translation
            transform.translation.x = float(cs_x(t))
            transform.translation.y = float(cs_y(t))
            transform.translation.z = float(cs_z(t))

            # 2. Rotation (Yaw towards center)
            current_yaw = float(cs_yaw(t))
            transform.rotation = self.get_quaternion_from_yaw(current_yaw)

            # 3. Velocity
            velocities = Twist()
            velocities.linear.x = float(cs_x(t, 1))
            velocities.linear.y = float(cs_y(t, 1))
            velocities.linear.z = float(cs_z(t, 1))

            # 4. Acceleration
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

        self.get_logger().info("Trajectory complete.")


def main(args=None):
    rclpy.init(args=args)
    node = SemicircleTrajectory()

    node.get_logger().info("Waiting for subscriber on /command/trajectory...")
    while node.trajectory_pub.get_subscription_count() == 0 and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1.0)

    node.get_logger().info(
        "Subscriber connected. Waiting for command on /mavros/setpoint_attitude/thrust..."
    )

    while rclpy.ok():
        if node.command_active:
            start = [0.0, 1.53]
            end = [0.0, -1.47]
            height = 1.65

            # Generate trajectory with Yaw data
            waypoints = node.generate_trajectory(
                start, end,
                height=height,
                straight_ratio=0.25
            )

            start_time = node.get_clock().now()
            node.start_trajectory(waypoints)
            duration = (node.get_clock().now() - start_time).nanoseconds / 1e9
            node.get_logger().info(f"One trajectory cycle took {duration:.2f} s.")

            time.sleep(1.0)
            node.command_active = False

        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("Script finished.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()