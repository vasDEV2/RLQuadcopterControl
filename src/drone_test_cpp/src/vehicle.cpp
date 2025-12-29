#include "drone_test_cpp/vehicle.hpp"



Vehicle::Vehicle(const rclcpp::Node::SharedPtr & node) : node_(node)
{
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);

    ctbr_pub_ = node_->create_publisher<px4_msgs::msg::VehicleRatesSetpoint>("/fmu/in/vehicle_rates_setpoint", 1);
    thrust_pub_ = node_->create_publisher<px4_msgs::msg::VehicleThrustSetpoint>("/fmu/in/vehicle_thrust_setpoint", 1);
    motor_pub_ = node_->create_publisher<px4_msgs::msg::ActuatorMotors>("/fmu/in/actuator_motors", 1);
    traj_pub_ = node_->create_publisher<px4_msgs::msg::TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 1);
    cmd_pub_ = node_->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 1);
    off_pub_ = node_->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", 10);

    odom_sub_ = node->create_subscription<px4_msgs::msg::VehicleOdometry>("/fmu/out/vehicle_odometry", qos,
            std::bind(&Vehicle::odom_callback, this, std::placeholders::_1));
    sens_sub_ = node->create_subscription<px4_msgs::msg::SensorCombined>("/fmu/out/sensor_combined", qos,
            std::bind(&Vehicle::imu_callback, this, std::placeholders::_1));
    status_sub_ = node->create_subscription<px4_msgs::msg::VehicleStatus>("/fmu/out/vehicle_status_v1", qos,
            std::bind(&Vehicle::status_callback, this, std::placeholders::_1));
    att_sub_ = node->create_subscription<px4_msgs::msg::VehicleAttitude>("/fmu/out/vehicle_attitude", qos,
            std::bind(&Vehicle::att_callback, this, std::placeholders::_1));


}

void Vehicle::change_mode(float mode){
    px4_msgs::msg::VehicleCommand msg{};
    msg.timestamp = node_->now().nanoseconds() / 1000;

    msg.command = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE;
    msg.param1 = 1.0;
    msg.param2 = static_cast<float>(mode);
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;

    cmd_pub_->publish(msg);
}

void Vehicle::arm_vehicle(){

    px4_msgs::msg::VehicleCommand msg{};

    msg.timestamp = int(node_->now().nanoseconds() / 1000);

    msg.command = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM;
    msg.param1 = 1.0;   
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;

    cmd_pub_->publish(msg);
}

void Vehicle::offboard_control(){

    px4_msgs::msg::OffboardControlMode offboard_msg{};
    offboard_msg.timestamp = node_->now().nanoseconds() / 1000;

    offboard_msg.position = false;
    offboard_msg.velocity = false;
    offboard_msg.acceleration = false;
    offboard_msg.attitude = false;
    offboard_msg.body_rate = false;
    offboard_msg.thrust_and_torque = false;
    offboard_msg.direct_actuator = true;
    // std::cout<<"PUBLISDHING";
    off_pub_->publish(offboard_msg);
}

void Vehicle::enable_offboard(){

    offboard_control();
    if(sp_count == 20){

        set_trajectory({0, 0, 0});
        arm_vehicle();
        RCLCPP_INFO(node_->get_logger(), "Vehicle ARMED!");
        offboard = true;
    }
    
    if(nav_state != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD){
        change_mode(6);
        set_trajectory({0, 0, 0});
    }

    set_trajectory({0,0,0});

    if(sp_count <= 21){
        sp_count++;
    }

}

void Vehicle::set_trajectory(Eigen::Vector3f pos){

    px4_msgs::msg::TrajectorySetpoint trajectory_msg{};

    // std::cout<<"POS Z: "<<pos[2];

    trajectory_msg.position[0] = pos[0];
    trajectory_msg.position[1] = pos[1];
    trajectory_msg.position[2] = pos[2];
    traj_pub_->publish(trajectory_msg);
}

void Vehicle::publish_motors(Eigen::Vector4f rates){

    px4_msgs::msg::ActuatorMotors msg{};

    msg.timestamp = node_->now().nanoseconds() / 1000;


    // msg.control = {rates[0], rates[2], rates[3], rates[1], 0., 0., 0.,0.,0.,0.,0.,0.};
    msg.control = {rates[0], rates[2], rates[3], rates[1], 0., 0., 0.,0.,0.,0.,0.,0.};


    motor_pub_->publish(msg);
}

void Vehicle::publish_ctbr(Eigen::Vector4f rates){

    px4_msgs::msg::VehicleRatesSetpoint msg{};

    msg.timestamp = node_->now().nanoseconds() / 1000;


    // msg.roll = rates[1];
    msg.roll = rates[1];
    // msg.pitch = rates[2];
    msg.pitch = rates[0];
    msg.yaw = rates[2];
    // msg.yaw = rates[3];
    // std::cout<<rates[0]<<std::endl;
    msg.thrust_body = {0.0, 0.0, -rates[0]};

    ctbr_pub_->publish(msg);

}


void Vehicle::odom_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg){

    pos << -msg->position[0]/10, msg->position[1]/10, -msg->position[2]/10;
    linear_velocity << -msg->velocity[0]/10, msg->velocity[1]/10, -msg->velocity[2]/10;

}

void Vehicle::imu_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg){

    angular_velocity << msg->gyro_rad[1], msg->gyro_rad[0], -msg->gyro_rad[2];
    linear_acceleration << msg->accelerometer_m_s2[0], msg->accelerometer_m_s2[1], msg->accelerometer_m_s2[2];

}

void Vehicle::status_callback(const px4_msgs::msg::VehicleStatus::SharedPtr msg){

    nav_state = msg->nav_state;
    arming_state = msg->arming_state;
}

void Vehicle::att_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg){

    // q << msg->q[0], msg->q[1], msg->q[2], msg->q[3];
    q.w() = msg->q[0];
    q.x() = msg->q[1];
    q.y() = msg->q[2];
    q.z() = msg->q[3];

}
    
