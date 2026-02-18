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

#include "franka_fr3_arm_controllers/joint_impedance_controller.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>

namespace franka_fr3_arm_controllers {
namespace {
constexpr double kTimestepSeconds = 0.001;

// Function to calculate joint velocity limits
std::pair<std::array<double, JointImpedanceController::kNumJoints>, std::array<double, JointImpedanceController::kNumJoints>>
calculate_joint_velocity_limits(const Eigen::Matrix<double, JointImpedanceController::kNumJoints, 1>& q) {
  std::array<double, JointImpedanceController::kNumJoints> q_dot_max;
  std::array<double, JointImpedanceController::kNumJoints> q_dot_min;

  // Calculate q_dot_max values
  q_dot_max[0] = std::min(
      2.62,
      std::max(0.0, -0.30 + sqrt(std::max(0.0, 12.0 * (2.75010 - q[0])))));
  q_dot_max[1] = std::min(
      2.62,
      std::max(0.0, -0.20 + sqrt(std::max(0.0, 5.17 * (1.79180 - q[1])))));
  q_dot_max[2] = std::min(
      2.62,
      std::max(0.0, -0.20 + sqrt(std::max(0.0, 7.00 * (2.90650 - q[2])))));
  q_dot_max[3] = std::min(
      2.62,
      std::max(0.0, -0.30 + sqrt(std::max(0.0, 8.00 * (-0.1458 - q[3])))));
  q_dot_max[4] = std::min(
      5.26,
      std::max(0.0, -0.35 + sqrt(std::max(0.0, 34.0 * (2.81010 - q[4])))));
  q_dot_max[5] = std::min(
      4.18,
      std::max(0.0, -0.35 + sqrt(std::max(0.0, 11.0 * (4.52050 - q[5])))));
  q_dot_max[6] = std::min(
      5.26,
      std::max(0.0, -0.35 + sqrt(std::max(0.0, 34.0 * (3.01960 - q[6])))));

  // Calculate q_dot_min values
  q_dot_min[0] = std::max(
      -2.62,
      std::min(0.0, 0.30 - sqrt(std::max(0.0, 12.0 * (2.750100 + q[0])))));
  q_dot_min[1] = std::max(
      -2.62,
      std::min(0.0, 0.20 - sqrt(std::max(0.0, 5.17 * (1.791800 + q[1])))));
  q_dot_min[2] = std::max(
      -2.62,
      std::min(0.0, 0.20 - sqrt(std::max(0.0, 7.00 * (2.906500 + q[2])))));
  q_dot_min[3] = std::max(
      -2.62,
      std::min(0.0, 0.30 - sqrt(std::max(0.0, 8.00 * (3.048100 + q[3])))));
  q_dot_min[4] = std::max(
      -5.26,
      std::min(0.0, 0.35 - sqrt(std::max(0.0, 34.0 * (2.810100 + q[4])))));
  q_dot_min[5] = std::max(
      -4.18,
      std::min(0.0, 0.35 - sqrt(std::max(0.0, 11.0 * (-0.54092 + q[5])))));
  q_dot_min[6] = std::max(
      -5.26,
      std::min(0.0, 0.35 - sqrt(std::max(0.0, 34.0 * (3.019600 + q[6])))));

  return {q_dot_min, q_dot_max};
}
}  // namespace

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" +
                           std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Joint state.
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" +
                           std::to_string(i) + "/position");
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" +
                           std::to_string(i) + "/velocity");
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" +
                           std::to_string(i) + "/effort");
  }

  // Robot state.
  for (const auto& name : robot_state_->get_state_interface_names()) {
    config.names.push_back(name);
  }

  // Robot model.
  for (const auto& franka_robot_model_name :
       franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }

  // Robot time.
  config.names.push_back(arm_id_ + "/robot_time");

  return config;
}

std::pair<std::array<double, JointImpedanceController::kNumJoints>, std::array<double, JointImpedanceController::kNumJoints>>
JointImpedanceController::ComputeReferencePositionAndVelocity(const std::array<double, JointImpedanceController::kNumJoints>& desired_position) {
  auto velocity_limits = calculate_joint_velocity_limits(last_position_);

  for (size_t i = 0; i < kNumJoints; ++i) {
    // Calculate the current error.
    double distance_to_target = desired_position[i] - last_position_[i];

    // Compute effective velocity that is scaled to satisfy acceleration limits.
    double min_velocity = std::max(
        velocity_limits.first[i] * velocity_limits_scaling_,
        last_velocity_[i] - acceleration_limits_[i] * kTimestepSeconds);
    double max_velocity = std::min(
        velocity_limits.second[i] * velocity_limits_scaling_,
        last_velocity_[i] + acceleration_limits_[i] * kTimestepSeconds);

    // Compute a velocity that is forced into limits.
    double clipped_velocity =
        std::min(std::max(distance_to_target / kTimestepSeconds, min_velocity),
                 max_velocity);
    double next_velocity = clipped_velocity;

    // Check if the velocity needs to be scaled down to not overshoot the
    // target. A stopping velocity is the effective speed required to reach the
    // target at maximum deceleration.
    if (distance_to_target >= 0.0) {
      const double stopping_velocity = std::sqrt(
          2.0 * std::abs(distance_to_target * acceleration_limits_[i]));
      if (clipped_velocity > stopping_velocity) {
        // Lower the speed by the maximum acceleration so can arrive at rest at
        // the target.
        next_velocity =
            last_velocity_[i] - acceleration_limits_[i] * kTimestepSeconds;
      }
    } else {
      const double stopping_velocity = -std::sqrt(
          2.0 * std::abs(distance_to_target * acceleration_limits_[i]));
      if (clipped_velocity < stopping_velocity) {
        // Increase the speed by the maximum acceleration so can arrive at rest
        // at the target.
        next_velocity =
            last_velocity_[i] + acceleration_limits_[i] * kTimestepSeconds;
      }
    }

    // Integrate the velocity to generate a kinematically consistent position.
    double next_position = last_position_[i] + next_velocity * kTimestepSeconds;

    last_velocity_[i] = next_velocity;
    last_position_[i] = next_position;
  }
  std::array<double, kNumJoints> position_arr;
  std::array<double, kNumJoints> velocity_arr;
  Eigen::Map<Eigen::Matrix<double, kNumJoints, 1>>(position_arr.data()) = last_position_;
  Eigen::Map<Eigen::Matrix<double, kNumJoints, 1>>(velocity_arr.data()) = last_velocity_;

  return std::make_pair(position_arr, velocity_arr);
}

std::array<double, JointImpedanceController::kNumJoints> JointImpedanceController::saturateTorqueRate(
    const std::array<double, JointImpedanceController::kNumJoints>& tau_d_calculated,
    const std::array<double, JointImpedanceController::kNumJoints>& tau_J_d) {
  std::array<double, kNumJoints> tau_d_saturated{};
  for (size_t i = 0; i < kNumJoints; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] +
        std::max(std::min(difference,
                          torque_derivative_limits_[i] * kTimestepSeconds),
                 -torque_derivative_limits_[i] * kTimestepSeconds);
  }
  return tau_d_saturated;
}

controller_interface::return_type JointImpedanceController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  // Start total timing
  auto start_total = std::chrono::high_resolution_clock::now();

  // Measure jitter (time between update calls)
  rclcpp::Time current_time = this->get_node()->now();
  if (last_update_time_.seconds() != 0.0) {
    double jitter_ms = (current_time - last_update_time_).seconds() * 1000.0;
    jitter_stats_.update(jitter_ms);
  }
  last_update_time_ = current_time;

  // Get the latest state.
  // Get the latest state.
  // robot_state_msg_ is a member variable, so we don't allocate here.
  // initialize_robot_state_msg ensures the message structure is correct (resizing vectors if needed, but usually they stay same size)
  robot_state_->initialize_robot_state_msg(robot_state_msg_);
  if (!robot_state_->get_values_as_message(robot_state_msg_)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Failed to get robot state as message.");
    return controller_interface::return_type::ERROR;
  }

  // Update desired position.
  std::array<double, kNumJoints> desired_position = *desired_position_.readFromRT();

  // Filter the measured velocity as it is a noisy signal.
  // Filter the measured velocity as it is a noisy signal.
  for (size_t i = 0; i < kNumJoints; i++) {
    dq_filtered_[i] = (1 - velocity_filter_alpha_) * dq_filtered_[i] +
                      velocity_filter_alpha_ *
                          robot_state_msg_.measured_joint_state.velocity[i];
  }

  std::array<double, kNumJoints> controller_output;
  // We currently support position setpoints. If we extend the interface to
  // support trajectories, we can call `SetTrajectoryReference`, providing
  // velocities and accelerations.
  auto reference = ComputeReferencePositionAndVelocity(desired_position);
  ref_pos_ = reference.first;
  ref_vel_ = reference.second;
  bool status = controller_.SetTrajectoryReference(ref_pos_, ref_vel_,
                                                   null_acc_);

  if (!status) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "DMJointPositionController: Failed to set PID reference.");
  }

  std::copy(robot_state_msg_.measured_joint_state.position.begin(),
            robot_state_msg_.measured_joint_state.position.end(),
            q_.begin());
  // dq_filtered_ is already updated in place above.

  status = controller_.ComputePIDOutput(q_, dq_filtered_, &pid_output_);
  if (!status) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "DMJointPositionController: Failed to compute PID output.");
  }
  controller_output = pid_output_;

  // Add in the coriolis term.
  std::array<double, kNumJoints> coriolis =
      franka_robot_model_->getCoriolisForceVector();
  for (size_t i = 0; i < kNumJoints; ++i) {
    controller_output[i] += coriolis[i];
  }

  // Apply saturation.
  std::array<double, kNumJoints> tau_J_d = {};
  for (size_t i = 0; i < kNumJoints; ++i) {
    tau_J_d[i] = robot_state_msg_.desired_joint_state.effort[i];
  }
  auto tau_d_saturated = saturateTorqueRate(controller_output, tau_J_d);
  std::copy(tau_d_saturated.begin(), tau_d_saturated.end(),
            last_torque_.begin());

  // Set command.
  for (size_t i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau_d_saturated[i]);
  }

  // End total timing
  auto end_total = std::chrono::high_resolution_clock::now();
  total_time_stats_.update(std::chrono::duration<double, std::milli>(end_total - start_total).count());

  update_counter_++;
  if (update_counter_ % 1000 == 0) {
    RCLCPP_INFO(get_node()->get_logger(),
                "Timing Stats (ms) [Min/Max/Avg]: "
                "Compute: %.3f / %.3f / %.3f | "
                "Jitter: %.3f / %.3f / %.3f",
                total_time_stats_.min_ms, total_time_stats_.max_ms, total_time_stats_.avg(),
                jitter_stats_.min_ms, jitter_stats_.max_ms, jitter_stats_.avg());
    
    total_time_stats_.reset();
    jitter_stats_.reset();
  }

  return controller_interface::return_type::OK;
}

void JointImpedanceController::jointStateCallback_(
    const sensor_msgs::msg::JointState& msg) {
  if (last_joint_state_time_.seconds() == 0.0) {
    return;
  }

  if (msg.position.size() < static_cast<size_t>(kNumJoints)) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received joint state size is smaller than expected size.");
    return;
  }
  std::array<double, kNumJoints> new_position;
  std::copy(msg.position.begin(),
            msg.position.begin() + kNumJoints,
            new_position.begin());
  desired_position_.writeFromNonRT(new_position);

  validateGelloPositions_(msg);
  last_joint_state_time_ = msg.header.stamp;
}

CallbackReturn JointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    if (arm_id_.empty()) {
      RCLCPP_FATAL(get_node()->get_logger(),
                   "arm_id parameter must be specified.");
      return CallbackReturn::ERROR;
    }
    auto_declare<std::vector<double>>("p_gains", std::vector<double>(kNumJoints, 0.0));
    auto_declare<std::vector<double>>("i_gains", std::vector<double>(kNumJoints, 0.0));
    auto_declare<std::vector<double>>("d_gains", std::vector<double>(kNumJoints, 0.0));
    auto_declare<double>("velocity_filter_alpha", 0.9);
    auto_declare<double>("velocity_limits_scaling", 0.9);
    auto_declare<std::vector<double>>("acceleration_limits",
                                      std::vector<double>(kNumJoints, 0.0));
    auto_declare<std::vector<double>>("torque_derivative_limits",
                                      std::vector<double>(kNumJoints, 0.0));
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n",
            e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  dq_filtered_.fill(0.0);

  // Namespace.
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    // Remove leading slash and add trailing underscore
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  // Robot model.
  franka_robot_model_ =
      std::make_unique<franka_semantic_components::FrankaRobotModel>(
          franka_semantic_components::FrankaRobotModel(
              arm_id_ + "/" + k_robot_model_interface_name,
              arm_id_ + "/" + k_robot_state_interface_name));

  // Robot description.
  auto parameters_client = std::make_shared<rclcpp::AsyncParametersClient>(
      get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();
  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Failed to get robot_description parameter.");
  }

  // Franka state.
  robot_state_ = std::make_unique<franka_semantic_components::FrankaRobotState>(
      franka_semantic_components::FrankaRobotState(
          arm_id_ + "/" + k_robot_state_interface_name, robot_description_));

  // Gains.
  auto p_gains = get_node()->get_parameter("p_gains").as_double_array();
  auto i_gains = get_node()->get_parameter("i_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();

  if (!validateGains_(p_gains, "p_gains") ||
      !validateGains_(i_gains, "i_gains") ||
      !validateGains_(d_gains, "d_gains")) {
    return CallbackReturn::FAILURE;
  }

  // Velocity filter and limits.
  velocity_filter_alpha_ =
      get_node()->get_parameter("velocity_filter_alpha").as_double();
  velocity_limits_scaling_ =
      get_node()->get_parameter("velocity_limits_scaling").as_double();

  // Acceleration limits.
  auto acceleration_limits =
      get_node()->get_parameter("acceleration_limits").as_double_array();
  if (acceleration_limits.size() == kNumJoints) {
    std::copy(acceleration_limits.begin(), acceleration_limits.end(),
              acceleration_limits_.begin());
  } else {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "acceleration_limits should be of size %d but is of size %ld",
                 kNumJoints, acceleration_limits.size());
    return CallbackReturn::FAILURE;
  }

  // Torque derivative limits.
  auto torque_derivative_limits =
      get_node()->get_parameter("torque_derivative_limits").as_double_array();
  if (torque_derivative_limits.size() == kNumJoints) {
    std::copy(torque_derivative_limits.begin(), torque_derivative_limits.end(),
              torque_derivative_limits_.begin());
  } else {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "torque_derivative_limits should be of size %d but is of size %ld",
        kNumJoints, torque_derivative_limits.size());
    return CallbackReturn::FAILURE;
  }

  // Subscribes to the topic that publishes the commands for the robot.
  joint_state_subscriber_ =
      get_node()->create_subscription<sensor_msgs::msg::JointState>(
          "gello/joint_states", 1,
          [this](const sensor_msgs::msg::JointState& msg) {
            jointStateCallback_(msg);
          });

  // Configure PID.
  std::array<double, kNumJoints> p_gains_arr;
  std::copy(p_gains.begin(), p_gains.end(), p_gains_arr.begin());
  if (!controller_.SetProportionalGains(p_gains_arr)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "DMJointPositionController: Failed to set P-gains.");
    return CallbackReturn::FAILURE;
  }

  std::array<double, kNumJoints> d_gains_arr;
  std::copy(d_gains.begin(), d_gains.end(), d_gains_arr.begin());
  if (!controller_.SetDerivativeGains(d_gains_arr)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "DMJointPositionController: Failed to set D-gains.");
    return CallbackReturn::FAILURE;
  }
  std::array<double, kNumJoints> i_gains_arr;
  std::copy(i_gains.begin(), i_gains.end(), i_gains_arr.begin());
  if (!controller_.SetIntegralGains(i_gains_arr)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "DMJointPositionController: Failed to set I-gains.");
    return CallbackReturn::FAILURE;
  }
  // Note: we currently do not set the saturation on the PID as the output will
  // go through additional modifications before being saturated. If adding
  // I-gains, please reconsider this choice as the anti-windup algorithm will
  // not be aware of the actual saturation.

  null_acc_.fill(0.0);

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  last_joint_state_time_ = get_node()->now();
  last_update_time_ = this->get_node()->now();

  // Get the latest state.
  robot_state_->assign_loaned_state_interfaces(state_interfaces_);
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  
  // Use member variable
  robot_state_->initialize_robot_state_msg(robot_state_msg_);
  if (!robot_state_->get_values_as_message(robot_state_msg_)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Failed to get robot state as message.");
    return CallbackReturn::ERROR;
  }

  // Save the state.
  std::array<double, kNumJoints> current_position;
  for (int i = 0; i < kNumJoints; ++i) {
    current_position[i] = robot_state_msg_.measured_joint_state.position[i];
    last_position_[i] = current_position[i];
    last_velocity_[i] = 0.0;
    last_torque_[i] = robot_state_msg_.desired_joint_state.effort[i];
    dq_filtered_[i] = 0.0;
  }
  desired_position_.initRT(current_position);

  std::copy(robot_state_msg_.measured_joint_state.position.begin(),
            robot_state_msg_.measured_joint_state.position.end(),
            q_.begin());
  if (!controller_.Reset(q_)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "DMJointPositionController: Failed to reset internal PID.");
  }

  return CallbackReturn::SUCCESS;
}

bool JointImpedanceController::validateGains_(const std::vector<double>& gains,
                                              const std::string& gains_name) {
  if (gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s parameter not set",
                 gains_name.c_str());
    return false;
  }

  if (gains.size() != static_cast<size_t>(kNumJoints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "%s should be of size %d but is of size %ld",
                 gains_name.c_str(), kNumJoints, gains.size());
    return false;
  }

  return true;
}



}  // namespace franka_fr3_arm_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::JointImpedanceController,
                       controller_interface::ControllerInterface)
