
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
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

    home_settle_sec_ = this->declare_parameter<double>(
      "home_settle_sec", 2.0);
    command_settle_sec_ = this->declare_parameter<double>(
      "command_settle_sec", 0.20);
    camera_ready_timeout_sec_ = this->declare_parameter<double>(
      "camera_ready_timeout_sec", 5.0);
    vision_wait_timeout_sec_ = this->declare_parameter<double>(
      "vision_wait_timeout_sec", 10.0);

    service_wait_timeout_sec_ = this->declare_parameter<double>(
      "service_wait_timeout_sec", 3.0);
    plan_timeout_sec_ = this->declare_parameter<double>(
      "plan_timeout_sec", 12.0);
    execute_timeout_sec_ = this->declare_parameter<double>(
      "execute_timeout_sec", 45.0);
    gripper_timeout_sec_ = this->declare_parameter<double>(
      "gripper_timeout_sec", 6.0);

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

    vision_sub_ =
      this->create_subscription<geometry_msgs::msg::PoseStamped>(
        vision_target_topic_,
        rclcpp::QoS(10),
        std::bind(
          &PickTaskNode::visionCallback,
          this,
          std::placeholders::_1),
        sub_options);

    target_cmd_pub_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>(
        motion_command_topic_,
        rclcpp::QoS(1).reliable());

    camera_trigger_pub_ = // 创建一个发布器，用于触发相机拍照
      this->create_publisher<std_msgs::msg::Empty>( // std_msgs::msg::Empty 是一个空消息类型，表示不需要传递任何数据，只是触发一个事件
        camera_trigger_topic_, // 发布到相机触发话题
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

    start_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/pick/start",
        std::bind(
          &PickTaskNode::startCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    status_srv_ =
      this->create_service<std_srvs::srv::Trigger>(
        "/pick/status",
        std::bind(
          &PickTaskNode::statusCallback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        rclcpp::ServicesQoS(),
        service_callback_group_);

    RCLCPP_INFO(
      this->get_logger(),
      "Pick flow: HOME -> CAMERA -> TARGET -> OPEN -> PREGRASP -> "
      "GRASP -> CLOSE -> LIFT -> HOME");
  }

private:
  void visionCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      latest_target_ = *msg;
      latest_target_rx_time_ =
        std::chrono::steady_clock::now();
      ++target_sequence_;
      have_target_ = true;
    }
    target_cv_.notify_all();
  }

  void statusCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    bool separate_mode = false;
    this->get_parameter(
      "use_separate_plan_execute", separate_mode);

    uint64_t seq = 0;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      seq = target_sequence_;
    }

    std::ostringstream oss;
    oss << "busy=" << (busy_.load() ? "true" : "false")
        << ", mode="
        << (separate_mode ? "PLAN_THEN_EXECUTE" : "DIRECT_MOVE")
        << ", target_seq=" << seq;

    response->success = !busy_.load();
    response->message = oss.str();
  }

  void startCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
      response->success = false;
      response->message = "Pick task is already running.";
      return;
    }

    struct BusyReset {
      std::atomic<bool>& flag;
      ~BusyReset() { flag.store(false); }
    } reset{busy_};

    std::string msg;
    std::string error;

    // 1. HOME
    if (!callTrigger(
        go_home_client_, go_home_service_,
        execute_timeout_sec_, msg))
    {
      response->success = false;
      response->message = "Initial HOME failed: " + msg;
      return;
    }

    if (home_settle_sec_ > 0.0) { // 回到Home点后停留home_settle_sec_的时间，确保机械臂稳定
      std::this_thread::sleep_for(
        std::chrono::duration<double>(home_settle_sec_));
    }

    // 2. 记录旧目标序号，然后触发一次拍照
    uint64_t baseline_sequence = 0;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      baseline_sequence = target_sequence_;
    }

    if (!publishCameraTrigger(error)) {
      response->success = false;
      response->message = "Camera trigger failed: " + error;
      return;
    }

    // 3. 等待拍照之后的新目标
    geometry_msgs::msg::PoseStamped vision_target;
    if (!waitForNewTarget(
        baseline_sequence, vision_target, error))
    {
      response->success = false;
      response->message =
        "Vision failed after camera trigger: " + error;
      return;
    }

    if (!required_frame_.empty() &&
        vision_target.header.frame_id != required_frame_)
    {
      response->success = false;
      response->message =
        "Vision target frame is '" +
        vision_target.header.frame_id +
        "', expected '" + required_frame_ + "'.";
      return;
    }

    geometry_msgs::msg::PoseStamped grasp = vision_target;
    grasp.pose.position.z += grasp_offset_z_;

    geometry_msgs::msg::PoseStamped pregrasp = grasp;
    pregrasp.pose.position.z += pregrasp_height_;

    geometry_msgs::msg::PoseStamped lift = grasp;
    lift.pose.position.z += lift_height_;

    RCLCPP_WARN(
      this->get_logger(),
      "Frozen target: grasp=(%.4f, %.4f, %.4f), "
      "pregrasp_z=%.4f, lift_z=%.4f",
      grasp.pose.position.x,
      grasp.pose.position.y,
      grasp.pose.position.z,
      pregrasp.pose.position.z,
      lift.pose.position.z);

    // 4. OPEN
    if (!callTrigger(
        gripper_open_client_, gripper_open_service_,
        gripper_timeout_sec_, msg))
    {
      response->success = false;
      response->message = "OPEN failed: " + msg;
      return;
    }

    // 5. PREGRASP
    if (!moveToPose(pregrasp, "PREGRASP", error)) {
      response->success = false;
      response->message = error;
      return;
    }

    if (!full_pick_enabled_) {
      response->success = true;
      response->message =
        "Stopped at PREGRASP because full_pick_enabled=false.";
      return;
    }

    // 6. GRASP
    if (!moveToPose(grasp, "GRASP", error)) {
      response->success = false;
      response->message = error;
      return;
    }

    // 7. CLOSE
    if (!callTrigger(
        gripper_close_client_, gripper_close_service_,
        gripper_timeout_sec_, msg))
    {
      response->success = false;
      response->message = "CLOSE failed: " + msg;
      return;
    }

    // 8. LIFT
    if (!moveToPose(lift, "LIFT", error)) {
      response->success = false;
      response->message =
        "Gripper closed, but LIFT failed: " + error;
      return;
    }

    // 9. RETURN HOME
    if (!callTrigger(
        go_home_client_, go_home_service_,
        execute_timeout_sec_, msg))
    {
      response->success = false;
      response->message =
        "Pick/Lift succeeded, but RETURN HOME failed: " + msg;
      return;
    }

    response->success = true;
    response->message =
      "Pick completed: HOME -> CAMERA -> PREGRASP -> "
      "GRASP -> CLOSE -> LIFT -> HOME.";
  }

  bool publishCameraTrigger(std::string& error)
  {
    const auto start = std::chrono::steady_clock::now();

    while (camera_trigger_pub_->get_subscription_count() == 0) { // 等待相机订阅者连接
      if (std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count()
          > camera_ready_timeout_sec_)
      {
        error = "No subscriber on " + camera_trigger_topic_;
        return false;
      }
      std::this_thread::sleep_for(50ms);
    }

    std_msgs::msg::Empty msg;
    camera_trigger_pub_->publish(msg); // 执行：发布一个空消息，触发相机拍照

    RCLCPP_INFO(
      this->get_logger(),
      "Published once to %s",
      camera_trigger_topic_.c_str());

    return true;
  }

  bool waitForNewTarget(
    uint64_t baseline_sequence,
    geometry_msgs::msg::PoseStamped& target,
    std::string& error)
  {
    std::unique_lock<std::mutex> lock(target_mutex_);

    const bool ok = target_cv_.wait_for(
      lock,
      std::chrono::duration<double>(vision_wait_timeout_sec_),
      [this, baseline_sequence]() {
        return have_target_ &&
               target_sequence_ > baseline_sequence;
      });

    if (!ok) {
      std::ostringstream oss;
      oss << "No NEW " << vision_target_topic_
          << " within " << vision_wait_timeout_sec_
          << " s after camera trigger.";
      error = oss.str();
      return false;
    }

    target = latest_target_;
    return true;
  }

  bool moveToPose(
    geometry_msgs::msg::PoseStamped target,
    const std::string& stage,
    std::string& error)
  {
    target.header.stamp = this->now();

    for (int i = 0; i < 3; ++i) {
      target_cmd_pub_->publish(target);
      std::this_thread::sleep_for(50ms);
    }

    if (command_settle_sec_ > 0.0) {
      std::this_thread::sleep_for(
        std::chrono::duration<double>(command_settle_sec_));
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
        error = stage + " planning failed: " + msg;
        return false;
      }

      if (!callTrigger(
          execute_client_, execute_service_,
          execute_timeout_sec_, msg))
      {
        error = stage + " execution failed: " + msg;
        return false;
      }

      return true;
    }

    if (!callTrigger(
        direct_move_client_, direct_move_service_,
        execute_timeout_sec_, msg))
    {
      error = stage + " direct move failed: " + msg;
      return false;
    }

    return true;
  }

  bool callTrigger(
    const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr& client,
    const std::string& service_name,
    double response_timeout_sec,
    std::string& message)
  {
    if (!client->wait_for_service(
        std::chrono::duration<double>(service_wait_timeout_sec_)))
    {
      message = "Service unavailable: " + service_name;
      return false;
    }

    auto request =
      std::make_shared<std_srvs::srv::Trigger::Request>();

    auto future = client->async_send_request(request);

    if (future.wait_for(
        std::chrono::duration<double>(response_timeout_sec))
        != std::future_status::ready)
    {
      message = "Timeout waiting for " + service_name;
      return false;
    }

    const auto response = future.get();
    message = response->message;
    return response->success;
  }

  std::string vision_target_topic_;
  std::string motion_command_topic_;
  std::string camera_trigger_topic_;

  std::string direct_move_service_;
  std::string plan_service_;
  std::string execute_service_;
  std::string go_home_service_;

  std::string gripper_open_service_;
  std::string gripper_close_service_;
  std::string required_frame_;

  double pregrasp_height_{0.08};
  double grasp_offset_z_{0.0};
  double lift_height_{0.10};

  double home_settle_sec_{0.50};
  double command_settle_sec_{0.20};
  double camera_ready_timeout_sec_{2.0};
  double vision_wait_timeout_sec_{4.0};

  double service_wait_timeout_sec_{3.0};
  double plan_timeout_sec_{12.0};
  double execute_timeout_sec_{45.0};
  double gripper_timeout_sec_{6.0};

  bool full_pick_enabled_{true};

  std::mutex target_mutex_;
  std::condition_variable target_cv_;
  bool have_target_{false};
  uint64_t target_sequence_{0};
  geometry_msgs::msg::PoseStamped latest_target_;
  std::chrono::steady_clock::time_point latest_target_rx_time_;

  std::atomic<bool> busy_{false};

  rclcpp::CallbackGroup::SharedPtr vision_callback_group_;
  rclcpp::CallbackGroup::SharedPtr client_callback_group_;
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;

  rclcpp::Subscription<
    geometry_msgs::msg::PoseStamped>::SharedPtr vision_sub_;
  rclcpp::Publisher<
    geometry_msgs::msg::PoseStamped>::SharedPtr target_cmd_pub_;
  rclcpp::Publisher<
    std_msgs::msg::Empty>::SharedPtr camera_trigger_pub_;

  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr direct_move_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr plan_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr execute_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr go_home_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr gripper_open_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr gripper_close_client_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr status_srv_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PickTaskNode>();

  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 4);

  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
