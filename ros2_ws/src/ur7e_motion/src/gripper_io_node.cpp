#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <stdexcept>

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "ur_msgs/msg/io_states.hpp"
#include "ur_msgs/srv/set_io.hpp"

using namespace std::chrono_literals;

class GripperIONode : public rclcpp::Node
{
public:
  GripperIONode()
  : Node("gripper_io_node")
  {
    output_pin_ = this->declare_parameter<int>("output_pin", 0);
    feedback_pin_ = this->declare_parameter<int>("feedback_pin", 0);
    open_output_state_ = this->declare_parameter<bool>("open_output_state", true);
    close_output_state_ = this->declare_parameter<bool>("close_output_state", false);
    feedback_active_state_ = this->declare_parameter<bool>("feedback_active_state", true);
    feedback_timeout_sec_ = this->declare_parameter<double>("feedback_timeout_sec", 6.0);
    feedback_idle_timeout_sec_ = this->declare_parameter<double>("feedback_idle_timeout_sec", 2.0);
    set_io_timeout_sec_ = this->declare_parameter<double>("set_io_timeout_sec", 2.0);
    io_states_topic_ = this->declare_parameter<std::string>(
      "io_states_topic", "/io_and_status_controller/io_states");
    set_io_service_ = this->declare_parameter<std::string>(
      "set_io_service", "/io_and_status_controller/set_io");

    if (output_pin_ < 0 || output_pin_ > 255 || feedback_pin_ < 0 || feedback_pin_ > 255) {
      throw std::runtime_error("output_pin / feedback_pin must be in [0, 255]");
    }
    if (feedback_timeout_sec_ <= 0.0 || feedback_idle_timeout_sec_ <= 0.0 || set_io_timeout_sec_ <= 0.0) {
      throw std::runtime_error("Timeout parameters must be > 0");
    }

    // Reentrant callbacks + MultiThreadedExecutor are intentional: the open/close
    // service waits for DI feedback while IO subscription/client callbacks must
    // continue to run on other executor threads.
    io_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    service_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = io_callback_group_;

    io_sub_ = this->create_subscription<ur_msgs::msg::IOStates>(
      io_states_topic_,
      rclcpp::QoS(20),
      std::bind(&GripperIONode::ioCallback, this, std::placeholders::_1),
      sub_options);

    set_io_client_ = this->create_client<ur_msgs::srv::SetIO>(
        set_io_service_,
        rclcpp::ServicesQoS(),
        io_callback_group_);

    open_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/gripper/open",
        std::bind(
            &GripperIONode::openCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    close_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/gripper/close",
        std::bind(
            &GripperIONode::closeCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    status_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/gripper/status",
        std::bind(
            &GripperIONode::statusCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    RCLCPP_INFO(
      this->get_logger(),
      "Gripper IO node ready: DO%d controls gripper, DI%d is completion feedback. "
      "OPEN=%s, CLOSE=%s, feedback active=%s",
      output_pin_, feedback_pin_,
      open_output_state_ ? "ON" : "OFF",
      close_output_state_ ? "ON" : "OFF",
      feedback_active_state_ ? "HIGH" : "LOW");
  }

private:
  struct ServiceAck
  {
    std::mutex mutex;
    std::condition_variable cv;
    bool done{false};
    bool success{false};
    std::string error;
  };

  void ioCallback(const ur_msgs::msg::IOStates::SharedPtr msg)
  {
    bool found_input = false;
    bool input_state = false;
    for (const auto & input : msg->digital_in_states) {
      if (static_cast<int>(input.pin) == feedback_pin_) {
        input_state = input.state;
        found_input = true;
        break;
      }
    }

    bool found_output = false;
    bool output_state = false;
    for (const auto & output : msg->digital_out_states) {
      if (static_cast<int>(output.pin) == output_pin_) {
        output_state = output.state;
        found_output = true;
        break;
      }
    }

    {
      std::lock_guard<std::mutex> lock(state_mutex_);

      if (found_input) {
        if (have_feedback_level_) {
          const bool became_active =
            (feedback_level_ != feedback_active_state_) &&
            (input_state == feedback_active_state_);

          if (became_active) {
            ++feedback_event_count_;
            RCLCPP_INFO(
              this->get_logger(),
              "DI%d completion pulse detected (event=%llu)",
              feedback_pin_,
              static_cast<unsigned long long>(feedback_event_count_));
          }
        }

        feedback_level_ = input_state;
        have_feedback_level_ = true;
      }

      if (found_output) {
        output_level_ = output_state;
        have_output_level_ = true;
      }
    }

    state_cv_.notify_all();
  }

  void openCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    commandAndWait(open_output_state_, "OPEN", response);
  }

  void closeCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    commandAndWait(close_output_state_, "CLOSE", response);
  }

  void statusCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    std::ostringstream oss;
    oss << "DO" << output_pin_ << "=";
    if (have_output_level_) {
      oss << (output_level_ ? "ON" : "OFF");
    } else {
      oss << "UNKNOWN";
    }

    oss << ", DI" << feedback_pin_ << "=";
    if (have_feedback_level_) {
      oss << (feedback_level_ ? "HIGH" : "LOW");
    } else {
      oss << "UNKNOWN";
    }

    oss << ", completion_events=" << feedback_event_count_;

    response->success = have_output_level_ && have_feedback_level_;
    response->message = oss.str();
  }

  bool waitForInitialIOState(std::string & error)
  {
    std::unique_lock<std::mutex> lock(state_mutex_);
    const auto timeout = std::chrono::duration<double>(set_io_timeout_sec_);

    const bool ready = state_cv_.wait_for(
      lock,
      timeout,
      [this]() {
        return have_feedback_level_ && have_output_level_;
      });

    if (!ready) {
      error = "No complete IO state received for DO" + std::to_string(output_pin_) +
        " / DI" + std::to_string(feedback_pin_);
      return false;
    }

    return true;
  }

  bool waitForFeedbackInactive(std::string & error)
  {
    std::unique_lock<std::mutex> lock(state_mutex_);

    if (!have_feedback_level_) {
      error = "DI feedback state is unknown";
      return false;
    }

    if (feedback_level_ != feedback_active_state_) {
      return true;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "DI%d is still in active feedback pulse; waiting for it to return idle...",
      feedback_pin_);

    const auto timeout = std::chrono::duration<double>(feedback_idle_timeout_sec_);
    const bool idle = state_cv_.wait_for(
      lock,
      timeout,
      [this]() {
        return have_feedback_level_ && (feedback_level_ != feedback_active_state_);
      });

    if (!idle) {
      error = "DI" + std::to_string(feedback_pin_) +
        " did not return to idle before command";
      return false;
    }

    return true;
  }

  bool setDigitalOutput(bool desired_state, std::string & error)
  {
    if (!set_io_client_->wait_for_service(std::chrono::duration<double>(set_io_timeout_sec_))) {
      error = "SetIO service unavailable: " + set_io_service_;
      return false;
    }

    auto request = std::make_shared<ur_msgs::srv::SetIO::Request>();
    request->fun = ur_msgs::srv::SetIO::Request::FUN_SET_DIGITAL_OUT;
    request->pin = static_cast<int8_t>(output_pin_);
    request->state = desired_state ?
      static_cast<float>(ur_msgs::srv::SetIO::Request::STATE_ON) :
      static_cast<float>(ur_msgs::srv::SetIO::Request::STATE_OFF);

    auto ack = std::make_shared<ServiceAck>();

    set_io_client_->async_send_request(
      request,
      [ack](rclcpp::Client<ur_msgs::srv::SetIO>::SharedFuture future) {
        std::lock_guard<std::mutex> lock(ack->mutex);
        try {
          const auto result = future.get();
          ack->success = result->success;
          if (!result->success) {
            ack->error = "UR SetIO service returned success=false";
          }
        } catch (const std::exception & ex) {
          ack->success = false;
          ack->error = ex.what();
        }
        ack->done = true;
        ack->cv.notify_all();
      });

    std::unique_lock<std::mutex> ack_lock(ack->mutex);
    const auto timeout = std::chrono::duration<double>(set_io_timeout_sec_);
    const bool completed = ack->cv.wait_for(
      ack_lock,
      timeout,
      [ack]() {return ack->done;});

    if (!completed) {
      error = "Timed out waiting for UR SetIO response";
      return false;
    }

    if (!ack->success) {
      error = ack->error.empty() ? "UR SetIO failed" : ack->error;
      return false;
    }

    return true;
  }

  void commandAndWait(
    bool desired_output_state,
    const std::string & action_name,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    // Prevent OPEN and CLOSE from running simultaneously.
    std::unique_lock<std::mutex> command_lock(command_mutex_, std::try_to_lock);
    if (!command_lock.owns_lock()) {
      response->success = false;
      response->message = "Another gripper command is already in progress.";
      return;
    }

    std::string error;

    if (!waitForInitialIOState(error)) {
      response->success = false;
      response->message = error;
      return;
    }

    if (!waitForFeedbackInactive(error)) {
      response->success = false;
      response->message = error;
      return;
    }

    uint64_t baseline_event_count = 0;
    bool output_already_matches = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      baseline_event_count = feedback_event_count_;
      output_already_matches = have_output_level_ && (output_level_ == desired_output_state);
    }

    // BY-P80 IO mode is level-controlled. Sending the same level again does not
    // create a new state transition, so no new completion pulse should be expected.
    if (output_already_matches) {
      response->success = true;
      response->message =
        "Gripper " + action_name + " already commanded: DO" +
        std::to_string(output_pin_) + " already " +
        (desired_output_state ? "ON" : "OFF") + ".";
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "%s command: setting DO%d=%s and waiting for a NEW DI%d completion pulse...",
      action_name.c_str(),
      output_pin_,
      desired_output_state ? "ON" : "OFF",
      feedback_pin_);

    if (!setDigitalOutput(desired_output_state, error)) {
      response->success = false;
      response->message = action_name + " failed: " + error;
      return;
    }

    std::unique_lock<std::mutex> state_lock(state_mutex_);
    const auto timeout = std::chrono::duration<double>(feedback_timeout_sec_);
    const bool feedback_received = state_cv_.wait_for(
      state_lock,
      timeout,
      [this, baseline_event_count]() {
        return feedback_event_count_ > baseline_event_count;
      });

    if (!feedback_received) {
      response->success = false;
      response->message =
        action_name + " command sent, but DI" + std::to_string(feedback_pin_) +
        " completion pulse timed out after " + std::to_string(feedback_timeout_sec_) + " s.";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    response->success = true;
    response->message =
      "Gripper " + action_name + " completed; DI" +
      std::to_string(feedback_pin_) + " completion pulse received.";

    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  }

  int output_pin_{0};
  int feedback_pin_{0};
  bool open_output_state_{true};
  bool close_output_state_{false};
  bool feedback_active_state_{true};
  double feedback_timeout_sec_{3.0};
  double feedback_idle_timeout_sec_{2.0};
  double set_io_timeout_sec_{2.0};
  std::string io_states_topic_;
  std::string set_io_service_;

  std::mutex state_mutex_;
  std::condition_variable state_cv_;
  bool have_feedback_level_{false};
  bool feedback_level_{false};
  bool have_output_level_{false};
  bool output_level_{false};
  uint64_t feedback_event_count_{0};

  std::mutex command_mutex_;

  rclcpp::CallbackGroup::SharedPtr io_callback_group_;
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;

  rclcpp::Subscription<ur_msgs::msg::IOStates>::SharedPtr io_sub_;
  rclcpp::Client<ur_msgs::srv::SetIO>::SharedPtr set_io_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr open_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr close_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr status_srv_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<GripperIONode>();

  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(),
    4);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
