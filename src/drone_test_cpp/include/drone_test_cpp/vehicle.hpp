#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"

#include "px4_msgs/msg/vehicle_rates_setpoint.hpp"
#include "px4_msgs/msg/vehicle_thrust_setpoint.hpp"
#include "px4_msgs/msg/actuator_motors.hpp"
#include "px4_msgs/msg/trajectory_setpoint.hpp"
#include "px4_msgs/msg/vehicle_command.hpp"
#include "px4_msgs/msg/offboard_control_mode.hpp"
#include "px4_msgs/msg/vehicle_odometry.hpp"
#include "px4_msgs/msg/sensor_combined.hpp"
#include "px4_msgs/msg/vehicle_status.hpp"
#include "px4_msgs/msg/vehicle_attitude.hpp"

class Vehicle
{
public:
  explicit Vehicle(const rclcpp::Node::SharedPtr &node);

  // State
  Eigen::Vector3f pos{Eigen::Vector3f::Zero()};
  Eigen::Vector3f linear_velocity{Eigen::Vector3f::Zero()};
  Eigen::Vector3f angular_velocity{Eigen::Vector3f::Zero()};
  Eigen::Vector3f linear_acceleration{Eigen::Vector3f::Zero()};
  Eigen::Quaternionf q;

  int nav_state{0};
  int arming_state{0};
  int sp_count{0};
  bool offboard{false};

  // Commands
  void change_mode(float mode);
  void arm_vehicle();
  void offboard_control();
  void enable_offboard();
  void set_trajectory(Eigen::Vector3f pos);
  void publish_motors(Eigen::Vector4f rates);
  void publish_ctbr(Eigen::Vector4f rates);

private:
  // ROS node handle
  rclcpp::Node::SharedPtr node_;

  // Publishers
  rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr ctbr_pub_;
  rclcpp::Publisher<px4_msgs::msg::VehicleThrustSetpoint>::SharedPtr thrust_pub_;
  rclcpp::Publisher<px4_msgs::msg::ActuatorMotors>::SharedPtr motor_pub_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr traj_pub_;
  rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr cmd_pub_;
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr off_pub_;

  // Subscriptions
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<px4_msgs::msg::SensorCombined>::SharedPtr sens_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr status_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr att_sub_;

  // Callbacks
  void odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
  void imu_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg);
  void status_callback(const px4_msgs::msg::VehicleStatus::SharedPtr msg);
  void att_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
};
