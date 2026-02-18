// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Eigen>
#include <array>
#include <limits>
#include "pid_controller.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_semantic_components/franka_robot_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_fr3_arm_controllers {

/**
 * Controller to move the robot to a desired joint position.
 */
class JointImpedanceController
    : public controller_interface::ControllerInterface {
 public:
  JointImpedanceController() : controller_(0.001) {}
  static constexpr int kNumJoints = 7;
  using Vector7d = Eigen::Matrix<double, kNumJoints, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;
  controller_interface::return_type update(
      const rclcpp::Time& time, const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

 private:
  // Saturates the torque rate to respect torque derivative limits.
  std::array<double, kNumJoints> saturateTorqueRate(
      const std::array<double, kNumJoints>& tau_d_calculated,
      const std::array<double, kNumJoints>& tau_J_d);
  std::pair<std::array<double, kNumJoints>, std::array<double, kNumJoints>>
  ComputeReferencePositionAndVelocity(const std::array<double, kNumJoints>& desired_position);
  bool validateGains_(const std::vector<double>& gains,
                      const std::string& gains_name);
  void jointStateCallback_(const sensor_msgs::msg::JointState& msg);

  std::string arm_id_;
  std::string namespace_prefix_;
  std::string robot_description_;
  Vector7d last_position_;
  Vector7d last_velocity_;
  Vector7d last_torque_;
  Vector7d q_;
  Vector7d dq_filtered_;
  Vector7d torque_derivative_limits_;
  Vector7d acceleration_limits_;
  double velocity_filter_alpha_;
  double velocity_limits_scaling_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      joint_state_subscriber_ = nullptr;
  realtime_tools::RealtimeBuffer<std::array<double, kNumJoints>> desired_position_;
  rclcpp::Time last_joint_state_time_;
  rclcpp::Time last_update_time_;
  std::unique_ptr<franka_semantic_components::FrankaRobotState> robot_state_;
  std::unique_ptr<franka_semantic_components::FrankaRobotModel>
      franka_robot_model_;
  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};
  gdm_robotics::PidController controller_;

  struct TimingStats {
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;
    int count = 0;

    void update(double ms) {
      if (ms < min_ms) min_ms = ms;
      if (ms > max_ms) max_ms = ms;
      sum_ms += ms;
      count++;
    }

    void reset() {
      min_ms = std::numeric_limits<double>::max();
      max_ms = 0.0;
      sum_ms = 0.0;
      count = 0;
    }

    double avg() const { return count > 0 ? sum_ms / count : 0.0; }
  };

  TimingStats total_time_stats_;
  TimingStats jitter_stats_;
  int update_counter_ = 0;
};

}  // namespace franka_fr3_arm_controllers
