#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "drone_test_cpp/vehicle.hpp"
#include "drone_test_cpp/model.hpp"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;   


class DroneTestNode : public rclcpp::Node
{   
public:

    DroneTestNode(const Eigen::Vector3f x) : Node("test_drone"),
    Gz(-9.81),
    mass(2.0),
    target(x),
    mean(Eigen::Vector4f::Ones()),
    std_(Eigen::Vector4f::Ones())
    {

        RCLCPP_INFO(this->get_logger(), "Node has Started");        
        mean *= (-Gz*mass)/4;
        std_ *= (-Gz*mass / 4)*0.1;
        float arm_l = 0.35;
        inertia = Eigen::Vector3f(0.021666667f, 0.021666667f, 0.04f).asDiagonal();
        kin_inv = Eigen::Vector3f(2.0f, 2.0f, 0.1f).asDiagonal();
        t_BM_ = arm_l * std::sqrt(0.5) * (Eigen::Matrix<float, 3, 4>() << 1, -1, -1, 1, -1, -1, 1, 1, 0, 0, 0, 0).finished();
        float kappa_ = 0.016;
        
        allocation_matrix.row(0) = Eigen::Vector4f::Ones();
        allocation_matrix.row(1) = t_BM_.row(0);  // roll
        allocation_matrix.row(2) = t_BM_.row(1);  // pitch
        // allocation_matrix.row(2) = t_BM_.row(1);
        allocation_matrix.row(3) = kappa_ * Eigen::Vector4f(1.0f, -1.0f, 1.0f, -1.0f);
        inverse_allocation = allocation_matrix.inverse();
        
        // target << 0.0, 0.0, 5.0;
        total_thrust = 9.4618;
        i = 0;

        // float timer_period_outer_loop = 0.02;
        timer_outer_loop_ = this->create_wall_timer(
            20ms, std::bind(&DroneTestNode::outer_callback, this));

        timer_inner_loop_ = this->create_wall_timer(
            1ms, std::bind(&DroneTestNode::inner_callback, this));
    }

    void init(){
        vehicle = std::make_unique<Vehicle>(shared_from_this());
        model = std::make_shared<ModelONNX>("/home/vasudevan/Desktop/flightmare/flightrl/examples/EpisodeV.onnx", 18);
    }

    rclcpp::TimerBase::SharedPtr timer_outer_loop_;
    rclcpp::TimerBase::SharedPtr timer_inner_loop_;
    std::unique_ptr<Vehicle> vehicle;
    std::shared_ptr<ModelONNX> model;
    Eigen::Vector3f target;
    int i;
    float Gz;
    float mass;
    float yaw_rate;
    Eigen::Vector4f mean;
    Eigen::Vector4f std_;
    float total_thrust;
    Eigen::Matrix3f inertia;
    Eigen::Matrix3f kin_inv;
    Eigen::Matrix<float, 3, 4> t_BM_;
    Eigen::Matrix4f allocation_matrix;
    Eigen::Matrix4f inverse_allocation;
    Eigen::VectorXf raw;

private:

    void inner_callback(){

        if(raw.size() == 0){
            return;
        }
        auto ac = raw;
        ac[0] = (ac[0] + 9.81)*mass;
        ac[1] = -ac[1];
        ac.segment(1, 2) *= 5.0f;

        auto omega_err = ac.segment(1, 3) - vehicle->angular_velocity;

        // std::cout<<"inert "<<inertia<<std::endl;
        // std::cout<<"kinv "<<kin_inv<<std::endl;
        // std::cout<<"t_BM "<<t_BM_<<std::endl;
        // std::cout<<"allocation  "<<allocation_matrix<<std::endl;
        // std::cout<<"inverse  "<<inverse_allocation<<std::endl;
        // std::cout<<"OMEGA ERR "<<omega_err<<std::endl;

        Eigen::Vector3f body_torque_des = inertia * kin_inv * omega_err + vehicle->angular_velocity.cross(inertia * vehicle->angular_velocity);

        Eigen::Vector4f thrust_and_torque(ac[0], body_torque_des[0], body_torque_des[1], 0.0f);

        // thrust_and_torque[1] = 0.2;
        // thrust_and_torque[2] = 0.0;
        thrust_and_torque[3] = 0.0;

        // std::cout<<"t and t: "<<thrust_and_torque<<std::endl;
        
        // thrust_and_torque[0] = std::sqrt(thrust_and_torque[0]);

        Eigen::Vector4f thrust = inverse_allocation * thrust_and_torque;

        thrust /= total_thrust;

        thrust = thrust.cwiseMax(0.0f).cwiseMin(1.0f);

        auto omega = thrust.array().sqrt().matrix();

        vehicle->publish_motors(omega);
        // std::cout<<"motor_cmd"<<omega<<std::endl;

    }

    void fly(){

        Eigen::Vector3f euler_zyx = vehicle->q.toRotationMatrix().eulerAngles(0, 1, 2);
        Eigen::Matrix3f R = vehicle->q.toRotationMatrix();
        // std::cout<<"quats :"<<vehicle->q<<std::endl;
        float temp;
        euler_zyx[0] = 0;
        // temp = euler_zyx[2];
        euler_zyx[2] = -euler_zyx[2];
        // euler_zyx[1] = temp;
        // Eigen::Matrix3f R = (Eigen::AngleAxisf(euler_zyx[0], Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf(euler_zyx[1], Eigen::Vector3f::UnitX())*Eigen::AngleAxisf(euler_zyx[2], Eigen::Vector3f::UnitY())).toRotationMatrix();
        Eigen::Vector<float, 9> rot_flat = Eigen::Map<Eigen::Vector<float, 9>>(R.data());
        // Eigen::VectorXf rot_flat = Eigen::Map<Eigen::VectorXf>(Eigen::Quaternionf(vehicle->q[0], vehicle->q[1], vehicle->q[2], vehicle->q[3]).toRotationMatrix().data(), 9);
        Eigen::VectorXf obs(19);
        obs << vehicle->pos, rot_flat, vehicle->linear_velocity, vehicle->angular_velocity, yaw_rate;
        // std::cout<<"OBS "<<obs<<std::endl;
        Eigen::VectorXf raw_action = model->predict(obs);   
        raw = raw_action.array().tanh().matrix();

        yaw_rate += euler_zyx[0]*0.02;
        // std::cout<<"ANG: "<<euler_zyx<<std::endl;
        // std::cout<<"ANG VEL: "<<vehicle->angular_velocity<<std::endl;
        std::cout<<"action_cmd: "<<raw<<std::endl;
        
        
    }

    // Eigen::Vector4f get_control(){
        
    //     Eigen::Vector3f goal_local = target - vehicle->pos;
    //     Eigen::VectorXf rot_flat = Eigen::Map<Eigen::VectorXf>(Eigen::Quaternionf(vehicle->q[0], vehicle->q[1], vehicle->q[2], vehicle->q[3]).toRotationMatrix().data(), 9);
    //     Eigen::VectorXf obs(18);
    //     obs << goal_local, rot_flat, vehicle->linear_velocity, vehicle->angular_velocity;

    //     // Eigen::Map<const Eigen::RowVectorXd> obs_inf(obs.data(), obs.size());

    //     Eigen::VectorXf raw_action = model->predict(obs);   
    //     auto raw = raw_action.array().tanh().matrix();
    //     // std::cout<<"START "<<raw<<"STOP"<<std::endl;
        
    //     Eigen::VectorXf action(4);

    //     action << raw[0]*std_[0] + mean[0], raw[1]*5, raw[2]*5, raw[3]*5;

    //     action[0] = action[0]/total_thrust;

    //     return action;

    // }

    // Eigen::Vector4f get_thrust()
    // {
    //     Eigen::Vector3f goal_local = target - vehicle->pos;
    //     // Eigen::VectorXf rot_flat = Eigen::Map<Eigen::VectorXf>(Eigen::Quaternionf(vehicle->q[0], vehicle->q[1], vehicle->q[2], vehicle->q[3]).toRotationMatrix().data(), 9);
    //     Eigen::Vector3f euler_zyx = Eigen::Quaternionf(vehicle->q[0], vehicle->q[1], vehicle->q[2], vehicle->q[3]).toRotationMatrix().eulerAngles(2, 1, 0);
        
    //     // Eigen::Vector3f euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
    //     Eigen::VectorXf obs(12);
    //     std::cout<<" E START: "<<euler_zyx<<" Estop"<<std::endl;
    //     obs << goal_local, euler_zyx, vehicle->linear_velocity, vehicle->angular_velocity;

    //     Eigen::Vector4f raw_action = model->predict(obs);   
    //     Eigen::Vector4f raw = raw_action.array().tanh().matrix();

    //     raw = raw.cwiseProduct(std_) + mean;

    //     raw =  raw / total_thrust;

    //     std::cout<<"START: "<<raw<<"STOP"<<std::endl;

    //     return raw;
        
    // }

    void outer_callback(){

        if(!vehicle->offboard){
            vehicle->enable_offboard();
            return;
        }

        // Eigen::Vector4f thrust = get_thrust();
        fly();
        vehicle->offboard_control();
        // vehicle->publish_motors(thrust);
        // vehicle->set_trajectory(target);
    }

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    Eigen::Vector3f target(0.0, 0.0, -5.0);

    auto node = std::make_shared<DroneTestNode>(target);
    node->init();

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}