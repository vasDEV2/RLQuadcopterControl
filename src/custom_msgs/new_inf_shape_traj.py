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

    def generate_exact_centered_infinity(self, p_start=np.array([1.5, 0.0, 1.0]), x0=1.0, y0=0.0, height=1.0, end_height=2.0, drift_x=0.0, drift_y=0.0, num_line=40, num_curve=400):        
        """
        start → P1 → full infinity loop centered at (0,0) → P1 → start

        Infinity:
        - center: (0,0)
        - start point: (x0, y0)
        - size automatically set by distance to origin
        """

        # ---------- Geometry ----------
        R = np.sqrt(x0**2 + y0**2)
        assert R > 1e-6, "Start point must not be at (0,0)"

        # Base infinity (starts at (0,R))
        t = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, num_curve)

        x = 0.9 * R * np.sin(2*t)    # = R*sin(t)*cos(t)
        y = R * np.sin(t)

        # Rotation so (0,R) → (x0,y0)
        phi = np.arctan2(y0, x0) - np.pi/2

        c, s = np.cos(phi), np.sin(phi)
        # xr = c*x - s*y
        # yr = s*x + c*y

        # z = np.linspace(height, end_height, num_curve)
        xr = c*x - s*y
        yr = s*x + c*y

        # Add smooth linear drift so the end point shifts without breaking the shape
        xr += np.linspace(0, drift_x, num_curve)
        yr += np.linspace(0, drift_y, num_curve)

        z = np.linspace(height, end_height, num_curve)

        seg2 = np.stack([xr, yr, z], axis=1)

        p1 = seg2[0]

        # ---------- Straight: start → P1 ----------
        t1 = np.linspace(0, 1, num_line)
        seg1 = p_start[None, :] + t1[:, None] * (p1 - p_start)

        # ---------- Straight: P1 → start ----------
        t3 = np.linspace(0, 1, num_line)
        p_end = seg2[-1]
        seg3 = p_end[None, :] + t3[:, None] * (p_start - p_end)

        return seg1, seg2, seg3
    
    def wait_for_enter(self, msg="Press ENTER to trigger next segment..."):
        self.get_logger().info(msg)
        input()

    def start_trajectory(self, waypoints, v_max = 0.3):
        """Generate and publish a smooth trajectory from waypoints."""
        # v_max = 0.3
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

            # print(transform.translation.x, transform.translation.y, transform.translation.z)

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
            a, b = 1.0, 0.8  # shape size
            # waypoints = node.generate_s_shape_mission(
            #     p_start=np.array([0.0, 0.0, 1.0]),
            #     p1=np.array([0.0, 1.0, 0.8]),
            #     p2=np.array([0.0, -1.0, 1.5]),
            #     s_amplitude=1.0
            # )


            seg1, seg2, seg3 = node.generate_exact_centered_infinity(
                p_start=np.array([0.0, 0.0, 1.3]), 
                x0=-0.1,
                y0=-1.12,
                height=1.13,
                end_height=1.28,
                drift_x=-0.33,  # Shifts the final X position by +0.2m over the loop
                drift_y=-0.23, # Shifts the final Y position by -0.1m over the loop
                num_line=40,
                num_curve=400
            )

            node.wait_for_enter("Press ENTER to start segment 1 (straight → P1)")
            node.start_trajectory(seg1, v_max=0.3)

            node.wait_for_enter("Press ENTER to start segment 2 (S-curve → P2)")
            node.start_trajectory(seg2, v_max=0.5)

            node.wait_for_enter("Press ENTER to start segment 3 (straight → start)")
            node.start_trajectory(seg3, v_max=0.3)

            # 🔁 Continuously run trajectory in loop
            start_time = node.get_clock().now()
            # node.start_trajectory(waypoints)
            duration = (node.get_clock().now() - start_time).nanoseconds / 1e9
            node.get_logger().info(f"One trajectory cycle took {duration:.2f} s.")
            # optional: small pause between cycles
            time.sleep(1.0)
            node.command_active = False

        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("Script finished.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

