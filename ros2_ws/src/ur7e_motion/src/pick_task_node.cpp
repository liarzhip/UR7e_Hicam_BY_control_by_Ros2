#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

class PickTaskNode : public rclcpp::Node
{
public:
  PickTaskNode()
  : Node("pick_task_node")
  {
    vision_target_topic_ = this->declare_parameter<std::string>(
      "vision_target_topic", "/vision/target_pose");
    vision_result_topic_ = this->declare_parameter<std::string>(
      "vision_result_topic", "/vision/result_status");
    motion_command_topic_ = this->declare_parameter<std::string>(
      "motion_command_topic", "/ur7e/target_pose_cmd");
    camera_trigger_topic_ = this->declare_parameter<std::string>(
      "camera_trigger_topic", "/hik_camera/run_once");

    direct_move_service_ = this->declare_parameter<std::string>(
      "direct_move_service", "/ur7e/move_to_target");
    plan_service_ = this->declare_parameter<std::string>(
      "plan_service", "/ur7e/plan_to_target");
    execute_service_ = this->declare_parameter<std::string>(
      "execute_service", "/ur7e/execute_last_plan");
    go_home_service_ = this->declare_parameter<std::string>(
      "go_home_service", "/ur7e/go_home");
    motion_stop_service_ = this->declare_parameter<std::string>(
      "motion_stop_service", "/ur7e/stop");

    gripper_open_service_ = this->declare_parameter<std::string>(
      "gripper_open_service", "/gripper/open");
    gripper_close_service_ = this->declare_parameter<std::string>(
      "gripper_close_service", "/gripper/close");

    required_frame_ = this->declare_parameter<std::string>(
      "required_frame", "base");

    pregrasp_height_ = this->declare_parameter<double>(
      "pregrasp_height", 0.08);
    grasp_offset_z_ = this->declare_parameter<double>(
      "grasp_offset_z", 0.0);
    lift_height_ = this->declare_parameter<double>(
      "lift_height", 0.10);

    // ------------------------------------------------------------
    // Sorting drop positions (Base frame, metres)
    // ------------------------------------------------------------
    drop_positions_configured_ = this->declare_parameter<bool>(
      "drop_positions_configured", false);

    bolt_drop_x_ = this->declare_parameter<double>("bolt_drop_x", 0.0);
    bolt_drop_y_ = this->declare_parameter<double>("bolt_drop_y", 0.0);
    bolt_drop_z_ = this->declare_parameter<double>("bolt_drop_z", 0.0);
    bolt_drop_qx_ = this->declare_parameter<double>("bolt_drop_qx", 0.0);
    bolt_drop_qy_ = this->declare_parameter<double>("bolt_drop_qy", 0.0);
    bolt_drop_qz_ = this->declare_parameter<double>("bolt_drop_qz", 0.0);
    bolt_drop_qw_ = this->declare_parameter<double>("bolt_drop_qw", 1.0);

    nut_drop_x_ = this->declare_parameter<double>("nut_drop_x", 0.0);
    nut_drop_y_ = this->declare_parameter<double>("nut_drop_y", 0.0);
    nut_drop_z_ = this->declare_parameter<double>("nut_drop_z", 0.0);
    nut_drop_qx_ = this->declare_parameter<double>("nut_drop_qx", 0.0);
    nut_drop_qy_ = this->declare_parameter<double>("nut_drop_qy", 0.0);
    nut_drop_qz_ = this->declare_parameter<double>("nut_drop_qz", 0.0);
    nut_drop_qw_ = this->declare_parameter<double>("nut_drop_qw", 1.0);

    drop_approach_height_ = this->declare_parameter<double>(
      "drop_approach_height", 0.10);
    drop_lowering_enabled_ = this->declare_parameter<bool>(
      "drop_lowering_enabled", true);

    // ------------------------------------------------------------
    // Timing / retry
    // ------------------------------------------------------------
    home_settle_sec_ = this->declare_parameter<double>(
      "home_settle_sec", 2.0);
    command_settle_sec_ = this->declare_parameter<double>(
      "command_settle_sec", 0.20);
    camera_ready_timeout_sec_ = this->declare_parameter<double>(
      "camera_ready_timeout_sec", 5.0);
    vision_wait_timeout_sec_ = this->declare_parameter<double>(
      "vision_wait_timeout_sec", 10.0);
    vision_retry_delay_sec_ = this->declare_parameter<double>(
      "vision_retry_delay_sec", 0.50);

    service_wait_timeout_sec_ = this->declare_parameter<double>(
      "service_wait_timeout_sec", 3.0);
    plan_timeout_sec_ = this->declare_parameter<double>(
      "plan_timeout_sec", 12.0);
    execute_timeout_sec_ = this->declare_parameter<double>(
      "execute_timeout_sec", 45.0);
    gripper_timeout_sec_ = this->declare_parameter<double>(
      "gripper_timeout_sec", 6.0);

    vision_retry_count_ = static_cast<int>(
      this->declare_parameter<int64_t>("vision_retry_count", 2));
    max_cycles_ = static_cast<int>(
      this->declare_parameter<int64_t>("max_cycles", 0));

    this->declare_parameter<bool>(
      "use_separate_plan_execute", false);

    full_pick_enabled_ = this->declare_parameter<bool>(
      "full_pick_enabled", true);

    vision_callback_group_ =
      this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    client_callback_group_ =
      this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    service_callback_group_ =
      this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = vision_callback_group_;

    vision_pose_sub_ =
      this->create_subscription<geometry_msgs::msg::PoseStamped>(
        vision_target_topic_,
        rclcpp::QoS(10),
        std::bind(
          &PickTaskNode::visionPoseCallback,
          this,
          std::placeholders::_1),
        sub_options);

    vision_result_sub_ =
      this->create_subscription<std_msgs::msg::String>(
        vision_result_topic_,
        rclcpp::QoS(10),
        std::bind(
          &PickTaskNode::visionResultCallback,
          this,
          std::placeholders::_1),
        sub_options);

    target_cmd_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(
        motion_command_topic_,
        rclcpp::QoS(1).reliable());

    camera_trigger_pub_ =
      this->create_publisher<std_msgs::msg::Empty>(
        camera_trigger_topic_,
        rclcpp::QoS(1).reliable());

    direct_move_client_ =
      this->create_client<std_srvs::srv::Trigger>(
        direct_move_service_,
        rclcpp::ServicesQoS(),
        client_callback_group_);

    plan_client_ =
      this->create_client<std_srvs::srv::Trigger>(
        plan_service_,
        rclcpp::ServicesQoS(),
        client_callback_group_);

    execute_client_ =
      this->create_client<std_srvs::srv::Trigger>(
        execute_service_,
        rclcpp::ServicesQoS(),
        client_callback_group_);

    go_home_client_ =
      this->create_client<std_srvs::srv::Trigger>(
        go_home_service_,
        rclcpp::ServicesQoS(),
        client_callback_group_);

    motion_stop_client_ =
      this->create_client<std_srvs::srv::Trigger>(
        motion_stop_service_,
        rclcpp::ServicesQoS(),
        client_callback_group_);

    gripper_open_client_ =
      this->create_client<std_srvs::srv::Trigger>(
        gripper_open_service_,
        rclcpp::ServicesQoS(),
        client_callback_group_);

    gripper_close_client_ =
      this->create_client<std_srvs::srv::Trigger>(
        gripper_close_service_,
        rclcpp::ServicesQoS(),
        client_callback_group_);

    // New continuous sorting interface.
    sort_start_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/sort/start",
        std::bind(
          &PickTaskNode::startCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    sort_stop_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/sort/stop",
        std::bind(
          &PickTaskNode::stopCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    sort_status_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/sort/status",
        std::bind(
          &PickTaskNode::statusCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    // Backward-compatible aliases. /pick/start now starts CONTINUOUS sorting.
    pick_start_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/pick/start",
        std::bind(
          &PickTaskNode::startCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    pick_stop_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/pick/stop",
        std::bind(
          &PickTaskNode::stopCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    pick_status_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/pick/status",
        std::bind(
          &PickTaskNode::statusCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    setState("IDLE", "READY");

    RCLCPP_INFO(
      this->get_logger(),
      "Continuous sorting flow: HOME -> CAMERA -> VISION -> PICK -> "
      "BOLT/NUT DROP -> HOME -> NEXT CYCLE; stop normally on result=none.");
  }

  ~PickTaskNode() override
  {
    stop_requested_.store(true);
    vision_cv_.notify_all();

    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

private:
  enum class VisionKind
  {
    TARGET,
    NONE,
    INVALID,
    TIMEOUT,
    STOPPED
  };

  struct VisionResult
  {
    VisionKind kind{VisionKind::INVALID};
    geometry_msgs::msg::PoseStamped pose;
    std::string class_name;
    std::string detail;
  };

  static std::string normalize(std::string value)
  {
    std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });

    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front()))) {
      value.erase(value.begin());
    }

    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back()))) {
      value.pop_back();
    }

    return value;
  }

  void visionPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      latest_target_ = *msg;
      ++target_sequence_;
    }
    vision_cv_.notify_all();
  }

  void visionResultCallback(
    const std_msgs::msg::String::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      latest_result_status_ = normalize(msg->data);
      ++result_sequence_;
    }
    vision_cv_.notify_all();
  }

  void startCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
      response->success = false;
      response->message = "Continuous sorting task is already running.";
      return;
    }

    if (full_pick_enabled_ && !drop_positions_configured_) {
      busy_.store(false);
      response->success = false;
      response->message =
        "Drop positions are not configured. Fill bolt/nut drop XYZ in "
        "motion.yaml and set drop_positions_configured=true.";
      return;
    }

    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }

    stop_requested_.store(false);
    cycle_count_.store(0);
    bolt_count_.store(0);
    nut_count_.store(0);

    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      current_class_.clear();
      stop_reason_.clear();
    }

    setState("STARTING", "");

    worker_thread_ =
      std::thread(&PickTaskNode::runSortingTask, this);

    response->success = true;
    response->message =
      "Continuous sorting task started. Use /sort/status to monitor.";
  }

  void stopCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    stop_requested_.store(true);
    vision_cv_.notify_all();
    setState("STOP_REQUESTED", "OPERATOR_REQUEST");

    // Best effort: also ask target_motion_node to cancel current MoveIt motion.
    if (motion_stop_client_->service_is_ready()) {
      auto request =
        std::make_shared<std_srvs::srv::Trigger::Request>();
      (void)motion_stop_client_->async_send_request(request);
    }

    response->success = true;
    response->message =
      "Stop requested. Current motion is also being asked to stop.";
  }

  void statusCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    bool separate_mode = false;
    this->get_parameter(
      "use_separate_plan_execute", separate_mode);

    std::string state;
    std::string current_class;
    std::string reason;

    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      state = state_;
      current_class = current_class_;
      reason = stop_reason_;
    }

    std::ostringstream oss;
    oss << "running=" << (busy_.load() ? "true" : "false")
        << ", state=" << state
        << ", mode="
        << (separate_mode ? "PLAN_THEN_EXECUTE" : "DIRECT_MOVE")
        << ", cycle=" << cycle_count_.load()
        << ", bolt=" << bolt_count_.load()
        << ", nut=" << nut_count_.load();

    if (!current_class.empty()) {
      oss << ", current_class=" << current_class;
    }

    if (!reason.empty()) {
      oss << ", stop_reason=" << reason;
    }

    response->success = true;
    response->message = oss.str();
  }

  void runSortingTask()
  {
    std::string msg;
    std::string error;

    // The task may be started from an arbitrary pose.
    setState("GO_HOME", "");
    if (!callTrigger(
        go_home_client_, go_home_service_,
        execute_timeout_sec_, msg))
    {
      finishTask("ERROR", "Initial HOME failed: " + msg);
      return;
    }

    while (rclcpp::ok() && !stop_requested_.load()) {
      // ----------------------------------------------------------
      // HOME -> wait stable -> one camera burst
      // ----------------------------------------------------------
      setState("HOME_SETTLE", "");

      if (!sleepInterruptible(home_settle_sec_)) {
        finishTask("STOPPED", "OPERATOR_REQUEST");
        return;
      }

      setState("WAIT_VISION", "");

      VisionResult vision =
        acquireVisionWithRetry(error);

      if (vision.kind == VisionKind::STOPPED) {
        finishTask("STOPPED", "OPERATOR_REQUEST");
        return;
      }

      if (vision.kind == VisionKind::NONE) {
        std::ostringstream done;
        done << "NO_TARGET; total=" << cycle_count_.load()
             << ", bolt=" << bolt_count_.load()
             << ", nut=" << nut_count_.load();

        RCLCPP_INFO(
          this->get_logger(),
          "Sorting completed normally: %s",
          done.str().c_str());

        finishTask("FINISHED", done.str());
        return;
      }

      if (vision.kind != VisionKind::TARGET) {
        finishTask(
          "ERROR",
          "Vision failed after retries: " + vision.detail);
        return;
      }

      if (!required_frame_.empty() &&
          vision.pose.header.frame_id != required_frame_)
      {
        finishTask(
          "ERROR",
          "Vision target frame is '" +
          vision.pose.header.frame_id +
          "', expected '" + required_frame_ + "'.");
        return;
      }

      const std::string target_class =
        normalize(vision.class_name);

      if (target_class != "bolt" &&
          target_class != "nut")
      {
        finishTask(
          "ERROR",
          "Unsupported final target class: " + target_class);
        return;
      }

      {
        std::lock_guard<std::mutex> lock(status_mutex_);
        current_class_ = target_class;
      }

      geometry_msgs::msg::PoseStamped grasp = vision.pose;
      grasp.pose.position.z += grasp_offset_z_;

      geometry_msgs::msg::PoseStamped pregrasp = grasp;
      pregrasp.pose.position.z += pregrasp_height_;

      geometry_msgs::msg::PoseStamped lift = grasp;
      lift.pose.position.z += lift_height_;

      RCLCPP_WARN(
        this->get_logger(),
        "Frozen target: class=%s, grasp=(%.4f, %.4f, %.4f), "
        "pregrasp_z=%.4f, lift_z=%.4f",
        target_class.c_str(),
        grasp.pose.position.x,
        grasp.pose.position.y,
        grasp.pose.position.z,
        pregrasp.pose.position.z,
        lift.pose.position.z);

      // ----------------------------------------------------------
      // PICK
      // ----------------------------------------------------------
      if (stop_requested_.load()) {
        finishTask("STOPPED", "OPERATOR_REQUEST");
        return;
      }

      setState("OPEN_GRIPPER", "");
      if (!callTrigger(
          gripper_open_client_, gripper_open_service_,
          gripper_timeout_sec_, msg))
      {
        finishTask("ERROR", "OPEN failed: " + msg);
        return;
      }

      setState("PREGRASP", "");
      if (!moveToPose(pregrasp, "PREGRASP", error)) {
        finishMotionAware(error);
        return;
      }

      if (!full_pick_enabled_) {
        finishTask(
          "DEBUG_STOPPED",
          "Stopped at PREGRASP because full_pick_enabled=false.");
        return;
      }

      setState("GRASP", "");
      if (!moveToPose(grasp, "GRASP", error)) {
        finishMotionAware(error);
        return;
      }

      setState("CLOSE_GRIPPER", "");
      if (!callTrigger(
          gripper_close_client_, gripper_close_service_,
          gripper_timeout_sec_, msg))
      {
        finishTask("ERROR", "CLOSE failed: " + msg);
        return;
      }

      setState("LIFT", "");
      if (!moveToPose(lift, "LIFT", error)) {
        finishMotionAware(
          "Gripper closed, but LIFT failed: " + error);
        return;
      }

      // ----------------------------------------------------------
      // SORTING DROP
      // ----------------------------------------------------------
      // Drop is a FIXED taught 6D pose in Base frame.
      // Do NOT inherit the grasp orientation from vision.
      geometry_msgs::msg::PoseStamped drop;
      drop.header.frame_id = required_frame_;

      if (target_class == "bolt") {
        drop.pose.position.x = bolt_drop_x_;
        drop.pose.position.y = bolt_drop_y_;
        drop.pose.position.z = bolt_drop_z_;

        drop.pose.orientation.x = bolt_drop_qx_;
        drop.pose.orientation.y = bolt_drop_qy_;
        drop.pose.orientation.z = bolt_drop_qz_;
        drop.pose.orientation.w = bolt_drop_qw_;
      } else {
        drop.pose.position.x = nut_drop_x_;
        drop.pose.position.y = nut_drop_y_;
        drop.pose.position.z = nut_drop_z_;

        drop.pose.orientation.x = nut_drop_qx_;
        drop.pose.orientation.y = nut_drop_qy_;
        drop.pose.orientation.z = nut_drop_qz_;
        drop.pose.orientation.w = nut_drop_qw_;
      }

      RCLCPP_WARN(
        this->get_logger(),
        "DROP TARGET [%s]: pos=(%.5f, %.5f, %.5f), "
        "q=(%.6f, %.6f, %.6f, %.6f)",
        target_class.c_str(),
        drop.pose.position.x,
        drop.pose.position.y,
        drop.pose.position.z,
        drop.pose.orientation.x,
        drop.pose.orientation.y,
        drop.pose.orientation.z,
        drop.pose.orientation.w);

      geometry_msgs::msg::PoseStamped drop_approach = drop;
      drop_approach.pose.position.z += drop_approach_height_;

      setState(
        target_class == "bolt"
          ? "MOVE_TO_BOLT_BIN"
          : "MOVE_TO_NUT_BIN",
        "");

      if (!moveToPose(
          drop_approach, "DROP_APPROACH", error))
      {
        finishMotionAware(error);
        return;
      }

      if (drop_lowering_enabled_) {
        setState("DROP_DESCEND", "");
        if (!moveToPose(drop, "DROP", error)) {
          finishMotionAware(error);
          return;
        }
      }

      setState("RELEASE", "");
      if (!callTrigger(
          gripper_open_client_, gripper_open_service_,
          gripper_timeout_sec_, msg))
      {
        finishTask("ERROR", "DROP OPEN failed: " + msg);
        return;
      }

      if (drop_lowering_enabled_) {
        setState("DROP_RETREAT", "");
        if (!moveToPose(
            drop_approach, "DROP_RETREAT", error))
        {
          finishMotionAware(error);
          return;
        }
      }

      // ----------------------------------------------------------
      // HOME -> next cycle
      // ----------------------------------------------------------
      setState("RETURN_HOME", "");
      if (!callTrigger(
          go_home_client_, go_home_service_,
          execute_timeout_sec_, msg))
      {
        finishTask(
          "ERROR",
          "Release succeeded, but RETURN HOME failed: " + msg);
        return;
      }

      const int cycle = cycle_count_.fetch_add(1) + 1;

      if (target_class == "bolt") {
        bolt_count_.fetch_add(1);
      } else {
        nut_count_.fetch_add(1);
      }

      RCLCPP_INFO(
        this->get_logger(),
        "Cycle %d completed. bolt=%d, nut=%d",
        cycle,
        bolt_count_.load(),
        nut_count_.load());

      {
        std::lock_guard<std::mutex> lock(status_mutex_);
        current_class_.clear();
      }

      if (max_cycles_ > 0 &&
          cycle >= max_cycles_)
      {
        finishTask(
          "FINISHED",
          "MAX_CYCLES reached: " +
          std::to_string(max_cycles_));
        return;
      }
    }

    finishTask("STOPPED", "OPERATOR_REQUEST");
  }

  VisionResult acquireVisionWithRetry(
    std::string& error)
  {
    VisionResult last_result;

    const int attempts =
      std::max(1, vision_retry_count_ + 1);

    for (int attempt = 1;
         attempt <= attempts;
         ++attempt)
    {
      if (stop_requested_.load()) {
        last_result.kind = VisionKind::STOPPED;
        last_result.detail = "Stop requested.";
        return last_result;
      }

      uint64_t baseline_result = 0;
      uint64_t baseline_target = 0;

      {
        std::lock_guard<std::mutex> lock(vision_mutex_);
        baseline_result = result_sequence_;
        baseline_target = target_sequence_;
      }

      if (!publishCameraTrigger(error)) {
        last_result.kind = VisionKind::INVALID;
        last_result.detail =
          "Camera trigger failed: " + error;
      } else {
        last_result =
          waitForVisionResult(
            baseline_result,
            baseline_target);
      }

      // TARGET and NONE are definitive outcomes.
      if (last_result.kind == VisionKind::TARGET ||
          last_result.kind == VisionKind::NONE ||
          last_result.kind == VisionKind::STOPPED)
      {
        return last_result;
      }

      if (attempt < attempts) {
        RCLCPP_WARN(
          this->get_logger(),
          "Vision attempt %d/%d failed (%s). Retrying at HOME...",
          attempt,
          attempts,
          last_result.detail.c_str());

        if (!sleepInterruptible(
            vision_retry_delay_sec_))
        {
          last_result.kind = VisionKind::STOPPED;
          last_result.detail = "Stop requested.";
          return last_result;
        }
      }
    }

    return last_result;
  }

  VisionResult waitForVisionResult(
    uint64_t baseline_result,
    uint64_t baseline_target)
  {
    VisionResult result;

    std::unique_lock<std::mutex> lock(vision_mutex_);

    const bool got_status =
      vision_cv_.wait_for(
        lock,
        std::chrono::duration<double>(
          vision_wait_timeout_sec_),
        [this, baseline_result]() {
          return stop_requested_.load() ||
                 result_sequence_ > baseline_result;
        });

    if (stop_requested_.load()) {
      result.kind = VisionKind::STOPPED;
      result.detail = "Stop requested.";
      return result;
    }

    if (!got_status) {
      result.kind = VisionKind::TIMEOUT;
      std::ostringstream oss;
      oss << "No NEW " << vision_result_topic_
          << " within " << vision_wait_timeout_sec_
          << " s after camera trigger.";
      result.detail = oss.str();
      return result;
    }

    const std::string status =
      normalize(latest_result_status_);

    if (status == "none") {
      result.kind = VisionKind::NONE;
      result.detail =
        "5-frame cycle confirmed no target.";
      return result;
    }

    if (status == "invalid") {
      result.kind = VisionKind::INVALID;
      result.detail =
        "Vision cycle returned invalid.";
      return result;
    }

    const std::string prefix = "target:";
    if (status.rfind(prefix, 0) != 0) {
      result.kind = VisionKind::INVALID;
      result.detail =
        "Unexpected vision result status: " + status;
      return result;
    }

    result.class_name =
      normalize(status.substr(prefix.size()));

    if (result.class_name != "bolt" &&
        result.class_name != "nut")
    {
      result.kind = VisionKind::INVALID;
      result.detail =
        "Unsupported target class in vision result: " +
        result.class_name;
      return result;
    }

    // Pose and status are published from the same final SVM cycle.
    // Because ROS callbacks are asynchronous, wait until this cycle's
    // Pose has also arrived.
    const bool got_pose =
      vision_cv_.wait_for(
        lock,
        std::chrono::duration<double>(
          vision_wait_timeout_sec_),
        [this, baseline_target]() {
          return stop_requested_.load() ||
                 target_sequence_ > baseline_target;
        });

    if (stop_requested_.load()) {
      result.kind = VisionKind::STOPPED;
      result.detail = "Stop requested.";
      return result;
    }

    if (!got_pose) {
      result.kind = VisionKind::INVALID;
      result.detail =
        "Received target class but no NEW target_pose "
        "from the same camera cycle.";
      return result;
    }

    result.pose = latest_target_;
    result.kind = VisionKind::TARGET;
    result.detail =
      "target:" + result.class_name;
    return result;
  }

  bool publishCameraTrigger(
    std::string& error)
  {
    const auto start =
      std::chrono::steady_clock::now();

    while (
      camera_trigger_pub_->get_subscription_count() == 0)
    {
      if (stop_requested_.load()) {
        error = "Stop requested.";
        return false;
      }

      if (std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count()
          > camera_ready_timeout_sec_)
      {
        error =
          "No subscriber on " + camera_trigger_topic_;
        return false;
      }

      std::this_thread::sleep_for(50ms);
    }

    std_msgs::msg::Empty msg;
    camera_trigger_pub_->publish(msg);

    RCLCPP_INFO(
      this->get_logger(),
      "Published once to %s",
      camera_trigger_topic_.c_str());

    return true;
  }

  bool moveToPose(
    geometry_msgs::msg::PoseStamped target,
    const std::string& stage,
    std::string& error)
  {
    if (stop_requested_.load()) {
      error = stage + " cancelled by stop request.";
      return false;
    }

    target.header.stamp = this->now();

    for (int i = 0; i < 3; ++i) {
      target_cmd_pub_->publish(target);
      std::this_thread::sleep_for(50ms);
    }

    if (!sleepInterruptible(command_settle_sec_)) {
      error = stage + " cancelled by stop request.";
      return false;
    }

    bool separate_mode = false;
    this->get_parameter(
      "use_separate_plan_execute", separate_mode);

    std::string msg;

    if (separate_mode) {
      if (!callTrigger(
          plan_client_, plan_service_,
          plan_timeout_sec_, msg))
      {
        error =
          stage + " planning failed: " + msg;
        return false;
      }

      if (!callTrigger(
          execute_client_, execute_service_,
          execute_timeout_sec_, msg))
      {
        error =
          stage + " execution failed: " + msg;
        return false;
      }

      return true;
    }

    if (!callTrigger(
        direct_move_client_, direct_move_service_,
        execute_timeout_sec_, msg))
    {
      error =
        stage + " direct move failed: " + msg;
      return false;
    }

    return true;
  }

  bool callTrigger(
    const rclcpp::Client<
      std_srvs::srv::Trigger>::SharedPtr& client,
    const std::string& service_name,
    double response_timeout_sec,
    std::string& message)
  {
    if (stop_requested_.load() &&
        service_name != motion_stop_service_)
    {
      message = "Stop requested.";
      return false;
    }

    if (!client->wait_for_service(
        std::chrono::duration<double>(
          service_wait_timeout_sec_)))
    {
      message =
        "Service unavailable: " + service_name;
      return false;
    }

    auto request =
      std::make_shared<
        std_srvs::srv::Trigger::Request>();

    auto future =
      client->async_send_request(request);

    if (future.wait_for(
        std::chrono::duration<double>(
          response_timeout_sec))
        != std::future_status::ready)
    {
      message =
        "Timeout waiting for " + service_name;
      return false;
    }

    const auto response = future.get();
    message = response->message;
    return response->success;
  }

  bool sleepInterruptible(
    double seconds)
  {
    if (seconds <= 0.0) {
      return !stop_requested_.load();
    }

    const auto end =
      std::chrono::steady_clock::now() +
      std::chrono::duration<double>(seconds);

    while (std::chrono::steady_clock::now() < end) {
      if (stop_requested_.load()) {
        return false;
      }

      std::this_thread::sleep_for(50ms);
    }

    return true;
  }

  void setState(
    const std::string& state,
    const std::string& reason)
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    state_ = state;

    if (!reason.empty()) {
      stop_reason_ = reason;
    }
  }

  void finishMotionAware(
    const std::string& error)
  {
    if (stop_requested_.load()) {
      finishTask("STOPPED", "OPERATOR_REQUEST");
    } else {
      finishTask("ERROR", error);
    }
  }

  void finishTask(
    const std::string& state,
    const std::string& reason)
  {
    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      state_ = state;
      stop_reason_ = reason;
      current_class_.clear();
    }

    busy_.store(false);

    if (state == "ERROR") {
      RCLCPP_ERROR(
        this->get_logger(),
        "Sorting task ended with ERROR: %s",
        reason.c_str());
    } else {
      RCLCPP_INFO(
        this->get_logger(),
        "Sorting task ended: state=%s, reason=%s",
        state.c_str(),
        reason.c_str());
    }
  }

  // Topics.
  std::string vision_target_topic_;
  std::string vision_result_topic_;
  std::string motion_command_topic_;
  std::string camera_trigger_topic_;

  // Services.
  std::string direct_move_service_;
  std::string plan_service_;
  std::string execute_service_;
  std::string go_home_service_;
  std::string motion_stop_service_;
  std::string gripper_open_service_;
  std::string gripper_close_service_;

  std::string required_frame_;

  // Pick geometry.
  double pregrasp_height_{0.08};
  double grasp_offset_z_{0.0};
  double lift_height_{0.10};

  // Drop geometry.
  bool drop_positions_configured_{false};
  double bolt_drop_x_{0.0};
  double bolt_drop_y_{0.0};
  double bolt_drop_z_{0.0};
  double bolt_drop_qx_{0.0};
  double bolt_drop_qy_{0.0};
  double bolt_drop_qz_{0.0};
  double bolt_drop_qw_{1.0};

  double nut_drop_x_{0.0};
  double nut_drop_y_{0.0};
  double nut_drop_z_{0.0};
  double nut_drop_qx_{0.0};
  double nut_drop_qy_{0.0};
  double nut_drop_qz_{0.0};
  double nut_drop_qw_{1.0};

  double drop_approach_height_{0.10};
  bool drop_lowering_enabled_{true};

  // Timing / retries.
  double home_settle_sec_{2.0};
  double command_settle_sec_{0.20};
  double camera_ready_timeout_sec_{5.0};
  double vision_wait_timeout_sec_{10.0};
  double vision_retry_delay_sec_{0.50};
  double service_wait_timeout_sec_{3.0};
  double plan_timeout_sec_{12.0};
  double execute_timeout_sec_{45.0};
  double gripper_timeout_sec_{6.0};

  int vision_retry_count_{2};
  int max_cycles_{0};
  bool full_pick_enabled_{true};

  // Vision synchronization.
  std::mutex vision_mutex_;
  std::condition_variable vision_cv_;
  uint64_t target_sequence_{0};
  uint64_t result_sequence_{0};
  geometry_msgs::msg::PoseStamped latest_target_;
  std::string latest_result_status_;

  // Task status.
  std::mutex status_mutex_;
  std::string state_{"IDLE"};
  std::string current_class_;
  std::string stop_reason_{"READY"};

  std::atomic<bool> busy_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<int> cycle_count_{0};
  std::atomic<int> bolt_count_{0};
  std::atomic<int> nut_count_{0};

  std::thread worker_thread_;

  rclcpp::CallbackGroup::SharedPtr vision_callback_group_;
  rclcpp::CallbackGroup::SharedPtr client_callback_group_;
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;

  rclcpp::Subscription<
    geometry_msgs::msg::PoseStamped>::SharedPtr vision_pose_sub_;
  rclcpp::Subscription<
    std_msgs::msg::String>::SharedPtr vision_result_sub_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseStamped>::SharedPtr target_cmd_pub_;
  rclcpp::Publisher<
    std_msgs::msg::Empty>::SharedPtr camera_trigger_pub_;

  rclcpp::Client<
    std_srvs::srv::Trigger>::SharedPtr direct_move_client_;
  rclcpp::Client<
    std_srvs::srv::Trigger>::SharedPtr plan_client_;
  rclcpp::Client<
    std_srvs::srv::Trigger>::SharedPtr execute_client_;
  rclcpp::Client<
    std_srvs::srv::Trigger>::SharedPtr go_home_client_;
  rclcpp::Client<
    std_srvs::srv::Trigger>::SharedPtr motion_stop_client_;
  rclcpp::Client<
    std_srvs::srv::Trigger>::SharedPtr gripper_open_client_;
  rclcpp::Client<
    std_srvs::srv::Trigger>::SharedPtr gripper_close_client_;

  rclcpp::Service<
    std_srvs::srv::Trigger>::SharedPtr sort_start_srv_;
  rclcpp::Service<
    std_srvs::srv::Trigger>::SharedPtr sort_stop_srv_;
  rclcpp::Service<
    std_srvs::srv::Trigger>::SharedPtr sort_status_srv_;

  rclcpp::Service<
    std_srvs::srv::Trigger>::SharedPtr pick_start_srv_;
  rclcpp::Service<
    std_srvs::srv::Trigger>::SharedPtr pick_stop_srv_;
  rclcpp::Service<
    std_srvs::srv::Trigger>::SharedPtr pick_status_srv_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node =
    std::make_shared<PickTaskNode>();

  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 4);

  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
