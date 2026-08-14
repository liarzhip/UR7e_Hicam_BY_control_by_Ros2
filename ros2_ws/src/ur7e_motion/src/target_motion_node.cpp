
#include <cmath>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>

using moveit::planning_interface::MoveGroupInterface;

template <typename T>
T get_or_declare_parameter(
    const rclcpp::Node::SharedPtr& node,
    const std::string& name,
    const T& default_value)
{
    if (!node->has_parameter(name)) {
        node->declare_parameter<T>(name, default_value);
    }
    T value;
    node->get_parameter(name, value);
    return value;
}

static std::string vector_to_yaml(const std::vector<double>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(8) << values[i];
    }
    oss << "]";
    return oss.str();
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>(
        "target_motion_node",
        rclcpp::NodeOptions()
            .automatically_declare_parameters_from_overrides(true));

    auto logger = node->get_logger();

    // 获取参数
    const std::string planning_group =
        get_or_declare_parameter<std::string>(
            node, "planning_group", "ur_manipulator");

    const double velocity_scale =
        get_or_declare_parameter<double>(
            node, "velocity_scale", 0.20);

    const double acceleration_scale =
        get_or_declare_parameter<double>(
            node, "acceleration_scale", 0.05);

    const double planning_time =
        get_or_declare_parameter<double>(
            node, "planning_time", 5.0);

    get_or_declare_parameter<bool>(
        node, "manual_plan_execute_enabled", true);

    std::vector<double> home_parameter =
        get_or_declare_parameter<std::vector<double>>(
            node, "home_joint_values", std::vector<double>{});

    MoveGroupInterface move_group(node, planning_group);
    move_group.setPlanningTime(planning_time);
    move_group.setMaxVelocityScalingFactor(velocity_scale);
    move_group.setMaxAccelerationScalingFactor(acceleration_scale);

    // Start MoveIt's internal CurrentStateMonitor early.
    // Use wait=0 here because the executor has not started spinning yet.
    // Once executor.spin() begins, /joint_states callbacks can populate it.
    move_group.startStateMonitor(0.0);

    RCLCPP_INFO(logger, "Planning frame: %s",
        move_group.getPlanningFrame().c_str());
    RCLCPP_INFO(logger, "End effector link: %s",
        move_group.getEndEffectorLink().c_str());

    geometry_msgs::msg::PoseStamped latest_target;
    bool have_target = false;
    std::shared_ptr<MoveGroupInterface::Plan> last_plan;

    std::vector<double> home_joint_values;
    bool have_home = false;

    std::mutex state_mutex;
    std::mutex moveit_mutex;
    std::mutex home_mutex;

    // IMPORTANT:
    // MoveGroupInterface's CurrentStateMonitor uses callbacks on this node.
    // If a service callback blocks while waiting for current state and both are
    // in the same MutuallyExclusive default callback group, the state callback
    // cannot run. Put our long-running service callbacks in their own Reentrant group.
    auto service_callback_group =
        node->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    auto create_trigger_service =
        [&](const std::string& name, auto&& callback)
        {
            return node->create_service<std_srvs::srv::Trigger>(
                name,
                std::forward<decltype(callback)>(callback),
                rclcpp::ServicesQoS(),
                service_callback_group);
        };

    const std::size_t joint_count = move_group.getJointNames().size();

    if (!home_parameter.empty()) {
        if (home_parameter.size() == joint_count) {
            std::lock_guard<std::mutex> lock(home_mutex);
            home_joint_values = home_parameter;
            have_home = true;
            RCLCPP_INFO(
                logger,
                "HOME loaded from parameter: %s",
                vector_to_yaml(home_joint_values).c_str());
        } else {
            RCLCPP_ERROR(
                logger,
                "home_joint_values has %zu values, but group has %zu joints.",
                home_parameter.size(), joint_count);
        }
    } else {
        RCLCPP_WARN(
            logger,
            "HOME not configured. Move robot to desired HOME and call /ur7e/set_home_here.");
    }

    auto target_sub =
        node->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/ur7e/target_pose_cmd",
            10,
            [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg)
            {
                if (msg->header.frame_id.empty()) {
                    RCLCPP_WARN(logger, "Received target without frame_id.");
                    return;
                }

                const auto& q = msg->pose.orientation;
                const double q_norm = std::sqrt(
                    q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);

                if (q_norm < 1e-6) {
                    RCLCPP_WARN(logger, "Invalid target quaternion.");
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    latest_target = *msg;
                    latest_target.pose.orientation.x /= q_norm;
                    latest_target.pose.orientation.y /= q_norm;
                    latest_target.pose.orientation.z /= q_norm;
                    latest_target.pose.orientation.w /= q_norm;
                    have_target = true;
                }

                RCLCPP_INFO(
                    logger,
                    "Target received: frame=%s x=%.4f y=%.4f z=%.4f",
                    msg->header.frame_id.c_str(),
                    msg->pose.position.x,
                    msg->pose.position.y,
                    msg->pose.position.z);
            });

    auto get_target = [&]() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return std::make_pair(have_target, latest_target);
    };

    auto set_pose_target =
        [&](const geometry_msgs::msg::PoseStamped& target,
            std::string& error)
        {
            move_group.setStartStateToCurrentState();
            move_group.clearPoseTargets();

            if (!move_group.setPoseTarget(target)) {
                error = "MoveIt rejected pose target.";
                return false;
            }
            return true;
        };

    auto move_service =
        create_trigger_service(
            "/ur7e/move_to_target",
            [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                auto [ok, target] = get_target();
                if (!ok) {
                    response->success = false;
                    response->message = "No target received.";
                    return;
                }

                std::lock_guard<std::mutex> move_lock(moveit_mutex);

                std::string error;
                if (!set_pose_target(target, error)) {
                    response->success = false;
                    response->message = error;
                    return;
                }

                RCLCPP_WARN(
                    logger,
                    "DIRECT MOVE: frame=%s x=%.4f y=%.4f z=%.4f",
                    target.header.frame_id.c_str(),
                    target.pose.position.x,
                    target.pose.position.y,
                    target.pose.position.z);

                const auto result = move_group.move();
                move_group.clearPoseTargets();

                response->success = static_cast<bool>(result);
                response->message = response->success ?
                    "Direct move succeeded." :
                    "MoveIt direct move failed.";
            });

    auto plan_service =
        create_trigger_service(
            "/ur7e/plan_to_target",
            [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                bool enabled = true;
                node->get_parameter("manual_plan_execute_enabled", enabled);

                if (!enabled) {
                    response->success = false;
                    response->message =
                        "Manual plan/execute disabled. Use /ur7e/move_to_target.";
                    return;
                }

                auto [ok, target] = get_target();
                if (!ok) {
                    response->success = false;
                    response->message = "No target received.";
                    return;
                }

                std::lock_guard<std::mutex> move_lock(moveit_mutex);

                std::string error;
                if (!set_pose_target(target, error)) {
                    response->success = false;
                    response->message = error;
                    return;
                }

                auto new_plan =
                    std::make_shared<MoveGroupInterface::Plan>();

                const auto result = move_group.plan(*new_plan);
                move_group.clearPoseTargets();

                if (!static_cast<bool>(result)) {
                    response->success = false;
                    response->message = "MoveIt planning failed.";
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    last_plan = new_plan;
                }

                response->success = true;
                response->message =
                    "Planning succeeded. Robot has NOT moved yet.";
            });

    auto execute_service =
        create_trigger_service(
            "/ur7e/execute_last_plan",
            [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                bool enabled = true;
                node->get_parameter("manual_plan_execute_enabled", enabled);

                if (!enabled) {
                    response->success = false;
                    response->message =
                        "Manual plan/execute disabled. Use /ur7e/move_to_target.";
                    return;
                }

                std::shared_ptr<MoveGroupInterface::Plan> plan_to_execute;
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    if (!last_plan) {
                        response->success = false;
                        response->message = "No valid plan available.";
                        return;
                    }
                    plan_to_execute = last_plan;
                }

                std::lock_guard<std::mutex> move_lock(moveit_mutex);
                const auto result = move_group.execute(*plan_to_execute);

                response->success = static_cast<bool>(result);
                response->message = response->success ?
                    "Trajectory execution succeeded." :
                    "Trajectory execution failed.";
            });

    auto set_home_service =
        create_trigger_service(
            "/ur7e/set_home_here",
            [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                std::vector<double> current;
                {
                    std::lock_guard<std::mutex> move_lock(moveit_mutex);

                    auto current_state = move_group.getCurrentState(2.0);
                    if (!current_state) {
                        response->success = false;
                        response->message =
                            "MoveIt CurrentStateMonitor did not provide a current state.";
                        return;
                    }

                    current_state->copyJointGroupPositions(
                        planning_group,
                        current);
                }

                if (current.size() != joint_count) {
                    response->success = false;
                    response->message =
                        "Current state received, but joint count does not match planning group.";
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(home_mutex);
                    home_joint_values = current;
                    have_home = true;
                }

                const std::string yaml = vector_to_yaml(current);

                response->success = true;
                response->message =
                    "HOME captured for this run. Put into motion.yaml: "
                    "home_joint_values: " + yaml;

                RCLCPP_WARN(
                    logger,
                    "HOME CAPTURED: home_joint_values: %s",
                    yaml.c_str());
            });

    auto home_status_service =
        create_trigger_service(
            "/ur7e/home_status",
            [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                std::lock_guard<std::mutex> lock(home_mutex);
                response->success = have_home;
                response->message = have_home ?
                    "HOME configured: " + vector_to_yaml(home_joint_values) :
                    "HOME is not configured.";
            });

    auto plan_home_service =
    create_trigger_service(
        "/ur7e/plan_home",
        [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
            std::shared_ptr<std_srvs::srv::Trigger::Response> response)
        {
            bool enabled = true;
            node->get_parameter(
                "manual_plan_execute_enabled",
                enabled);

            if (!enabled)
            {
                response->success = false;
                response->message =
                    "Manual plan/execute is disabled.";
                return;
            }

            std::vector<double> home;

            {
                std::lock_guard<std::mutex> lock(home_mutex);

                if (!have_home)
                {
                    response->success = false;
                    response->message =
                        "HOME is not configured.";
                    return;
                }

                home = home_joint_values;
            }

            std::lock_guard<std::mutex> move_lock(
                moveit_mutex);

            /*
             * 从机器人真实当前位置开始
             */
            move_group.setStartStateToCurrentState();

            move_group.clearPoseTargets();

            /*
             * HOME 是关节空间目标
             */
            const bool accepted =
                move_group.setJointValueTarget(home);

            if (!accepted)
            {
                response->success = false;
                response->message =
                    "MoveIt rejected HOME joint target.";
                return;
            }

            /*
             * 只 Plan，不执行
             */
            auto new_plan =
                std::make_shared<
                    MoveGroupInterface::Plan>();

            const auto result =
                move_group.plan(*new_plan);

            if (!static_cast<bool>(result))
            {
                response->success = false;
                response->message =
                    "HOME planning failed.";

                RCLCPP_ERROR(
                    logger,
                    "HOME planning failed.");

                return;
            }

            /*
             * 保存给 /execute_last_plan
             */
            {
                std::lock_guard<std::mutex> lock(
                    state_mutex);

                last_plan = new_plan;
            }

            response->success = true;

            response->message =
                "HOME planning succeeded. "
                "Robot has NOT moved. "
                "Call /ur7e/execute_last_plan.";

            RCLCPP_INFO(
                logger,
                "HOME planning succeeded. "
                "Waiting for execute.");
        });

    auto go_home_service =
        create_trigger_service(
            "/ur7e/go_home",
            [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                std::vector<double> home;
                {
                    std::lock_guard<std::mutex> lock(home_mutex);
                    if (!have_home) {
                        response->success = false;
                        response->message =
                            "HOME is not configured. Call /ur7e/set_home_here first.";
                        return;
                    }
                    home = home_joint_values;
                }

                std::lock_guard<std::mutex> move_lock(moveit_mutex);

                move_group.setStartStateToCurrentState();
                move_group.clearPoseTargets();

                if (!move_group.setJointValueTarget(home)) {
                    response->success = false;
                    response->message = "MoveIt rejected HOME joint target.";
                    return;
                }

                RCLCPP_WARN(
                    logger,
                    "GO HOME: %s",
                    vector_to_yaml(home).c_str());

                const auto result = move_group.move();

                response->success = static_cast<bool>(result);
                response->message = response->success ?
                    "HOME reached." :
                    "Failed to move HOME.";
            });

    auto stop_service =
        create_trigger_service(
            "/ur7e/stop",
            [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                std::lock_guard<std::mutex> move_lock(moveit_mutex);
                move_group.stop();
                response->success = true;
                response->message = "MoveIt stop requested.";
            });

    RCLCPP_INFO(logger, "UR7e target motion node READY");
    RCLCPP_INFO(logger, "  /ur7e/move_to_target");
    RCLCPP_INFO(logger, "  /ur7e/go_home");
    RCLCPP_INFO(logger, "  /ur7e/set_home_here");
    RCLCPP_INFO(logger, "  /ur7e/home_status");
    RCLCPP_INFO(logger, "  /ur7e/plan_to_target");
    RCLCPP_INFO(logger, "  /ur7e/execute_last_plan");
    RCLCPP_INFO(logger, "  /ur7e/stop");

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
