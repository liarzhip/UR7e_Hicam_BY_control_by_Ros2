#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <std_srvs/srv/trigger.hpp>

#include <moveit/move_group_interface/move_group_interface.hpp>

using moveit::planning_interface::MoveGroupInterface; // 直接使用MoveGroupInterface名字


template <typename T> //
T get_or_declare_parameter(
    const rclcpp::Node::SharedPtr& node,
    const std::string& name,
    const T& default_value)
{
    if (!node->has_parameter(name))
    {
        node->declare_parameter<T>(name, default_value);
    }

    T value;
    node->get_parameter(name, value);

    return value;
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    /*
     * MoveIt 需要 automatically_declare_parameters_from_overrides(true)
     * 来读取 robot_description 等参数。
     */
    auto node = std::make_shared<rclcpp::Node>(
        "target_motion_node",
        rclcpp::NodeOptions()
            .automatically_declare_parameters_from_overrides(true)
    );

    auto logger = node->get_logger();


    /*
     * -------------------------------
     * 参数
     * -------------------------------
     */

    const std::string planning_group =
        get_or_declare_parameter<std::string>(
            node,
            "planning_group",
            "ur_manipulator"
        );

    const double pregrasp_offset_z =
        get_or_declare_parameter<double>(
            node,
            "pregrasp_offset_z",
            0.10
        );

    const double velocity_scale =
        get_or_declare_parameter<double>(
            node,
            "velocity_scale",
            0.05
        );

    const double acceleration_scale =
        get_or_declare_parameter<double>(
            node,
            "acceleration_scale",
            0.05
        );

    const double planning_time =
        get_or_declare_parameter<double>(
            node,
            "planning_time",
            5.0
        );


    /*
     * -------------------------------
     * 创建 MoveGroupInterface
     * -------------------------------
     */

    RCLCPP_INFO(
        logger,
        "Creating MoveGroupInterface, group = %s",
        planning_group.c_str()
    );

    MoveGroupInterface move_group(
        node,
        planning_group
    );


    /*
     * 设置规划参数
     */

    move_group.setPlanningTime(planning_time);

    move_group.setMaxVelocityScalingFactor(
        velocity_scale
    );

    move_group.setMaxAccelerationScalingFactor(
        acceleration_scale
    );


    RCLCPP_INFO(
        logger,
        "Planning frame: %s",
        move_group.getPlanningFrame().c_str()
    );

    RCLCPP_INFO(
        logger,
        "End effector link: %s",
        move_group.getEndEffectorLink().c_str()
    );


    /*
     * -------------------------------
     * 保存视觉目标
     * -------------------------------
     */

    geometry_msgs::msg::PoseStamped latest_target;

    bool have_target = false;

    std::mutex state_mutex;

    std::mutex moveit_mutex;


    /*
     * 保存最后一次规划
     */

    std::shared_ptr<
        MoveGroupInterface::Plan
    > last_plan;


    /*
     * -------------------------------
     * Subscribe:
     *
     * /vision/target_pose
     * -------------------------------
     */

    auto target_sub =
        node->create_subscription<
            geometry_msgs::msg::PoseStamped
        >(
            "/vision/target_pose",
            10,

            [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg)
            {
                /*
                 * frame_id 必须存在
                 */

                if (msg->header.frame_id.empty())
                {
                    RCLCPP_WARN(
                        logger,
                        "Received target without frame_id."
                    );

                    return;
                }


                /*
                 * 检查四元数是否合法
                 */

                const auto& q = msg->pose.orientation;

                const double q_norm =
                    std::sqrt(
                        q.x * q.x +
                        q.y * q.y +
                        q.z * q.z +
                        q.w * q.w
                    );


                if (q_norm < 1e-6)
                {
                    RCLCPP_WARN(
                        logger,
                        "Invalid target quaternion."
                    );

                    return;
                }


                /*
                 * 保存最新目标
                 */

                {
                    std::lock_guard<std::mutex> lock(
                        state_mutex
                    );

                    latest_target = *msg;

                    /*
                     * 自动归一化
                     */

                    latest_target.pose.orientation.x /= q_norm;
                    latest_target.pose.orientation.y /= q_norm;
                    latest_target.pose.orientation.z /= q_norm;
                    latest_target.pose.orientation.w /= q_norm;

                    have_target = true;
                }


                RCLCPP_INFO(
                    logger,
                    "Target received: frame=%s, x=%.4f, y=%.4f, z=%.4f",
                    msg->header.frame_id.c_str(),
                    msg->pose.position.x,
                    msg->pose.position.y,
                    msg->pose.position.z
                );
            }
        );


    /*
     * -------------------------------
     * Service 1:
     *
     * /ur7e/plan_to_target
     *
     * 只规划，不执行
     * -------------------------------
     */

    auto plan_service =
        node->create_service<std_srvs::srv::Trigger>(
            "/ur7e/plan_to_target",

            [&](const std::shared_ptr<
                    std_srvs::srv::Trigger::Request> /*request*/,
                std::shared_ptr<
                    std_srvs::srv::Trigger::Response> response)
            {
                geometry_msgs::msg::PoseStamped target;

                /*
                 * 读取当前视觉目标
                 */

                {
                    std::lock_guard<std::mutex> lock(
                        state_mutex
                    );

                    if (!have_target)
                    {
                        response->success = false;

                        response->message =
                            "No vision target received.";

                        return;
                    }

                    target = latest_target;
                }


                /*
                 * 当前先做 PRE-GRASP
                 *
                 * 即目标上方 pregrasp_offset_z
                 */

                target.pose.position.z +=
                    pregrasp_offset_z;


                RCLCPP_INFO(
                    logger,
                    "Planning target:"
                    " frame=%s x=%.4f y=%.4f z=%.4f",
                    target.header.frame_id.c_str(),
                    target.pose.position.x,
                    target.pose.position.y,
                    target.pose.position.z
                );


                /*
                 * 防止同时调用 plan / execute
                 */

                std::lock_guard<std::mutex> move_lock(
                    moveit_mutex
                );


                /*
                 * 从真实机器人当前状态开始规划
                 */

                move_group.setStartStateToCurrentState();

                move_group.clearPoseTargets();


                /*
                 * 设置末端目标
                 */

                const bool target_ok =
                    move_group.setPoseTarget(
                        target
                    );


                if (!target_ok)
                {
                    response->success = false;

                    response->message =
                        "MoveIt rejected pose target.";

                    return;
                }


                /*
                 * 创建轨迹规划
                 */

                auto new_plan =
                    std::make_shared<
                        MoveGroupInterface::Plan
                    >();


                const auto result =
                    move_group.plan(
                        *new_plan
                    );


                move_group.clearPoseTargets();


                /*
                 * 规划失败
                 */

                if (!static_cast<bool>(result))
                {
                    response->success = false;

                    response->message =
                        "MoveIt planning failed.";

                    RCLCPP_ERROR(
                        logger,
                        "Planning failed."
                    );

                    return;
                }


                /*
                 * 保存轨迹
                 */

                {
                    std::lock_guard<std::mutex> lock(
                        state_mutex
                    );

                    last_plan = new_plan;
                }


                response->success = true;

                response->message =
                    "Planning succeeded. "
                    "Robot has NOT moved yet.";


                RCLCPP_INFO(
                    logger,
                    "Planning succeeded. "
                    "Waiting for execute command."
                );
            }
        );


    /*
     * -------------------------------
     * Service 2:
     *
     * /ur7e/execute_last_plan
     * -------------------------------
     */

    auto execute_service =
        node->create_service<std_srvs::srv::Trigger>(
            "/ur7e/execute_last_plan",

            [&](const std::shared_ptr<
                    std_srvs::srv::Trigger::Request> /*request*/,
                std::shared_ptr<
                    std_srvs::srv::Trigger::Response> response)
            {
                std::shared_ptr<
                    MoveGroupInterface::Plan
                > plan_to_execute;


                {
                    std::lock_guard<std::mutex> lock(
                        state_mutex
                    );

                    if (!last_plan)
                    {
                        response->success = false;

                        response->message =
                            "No valid plan available.";

                        return;
                    }

                    plan_to_execute = last_plan;
                }


                RCLCPP_WARN(
                    logger,
                    "Executing last planned trajectory."
                );


                std::lock_guard<std::mutex> move_lock(
                    moveit_mutex
                );


                const auto result =
                    move_group.execute(
                        *plan_to_execute
                    );


                if (!static_cast<bool>(result))
                {
                    response->success = false;

                    response->message =
                        "Trajectory execution failed.";

                    return;
                }


                response->success = true;

                response->message =
                    "Trajectory execution succeeded.";


                RCLCPP_INFO(
                    logger,
                    "Trajectory finished."
                );
            }
        );


    /*
     * -------------------------------
     * Service 3:
     *
     * /ur7e/stop
     * -------------------------------
     */

    auto stop_service =
        node->create_service<std_srvs::srv::Trigger>(
            "/ur7e/stop",

            [&](const std::shared_ptr<
                    std_srvs::srv::Trigger::Request> /*request*/,
                std::shared_ptr<
                    std_srvs::srv::Trigger::Response> response)
            {
                std::lock_guard<std::mutex> move_lock(
                    moveit_mutex
                );

                move_group.stop();

                response->success = true;

                response->message =
                    "MoveIt stop requested.";

                RCLCPP_WARN(
                    logger,
                    "STOP requested."
                );
            }
        );


    RCLCPP_INFO(
        logger,
        "======================================"
    );

    RCLCPP_INFO(
        logger,
        "UR7e target motion node READY"
    );

    RCLCPP_INFO(
        logger,
        "Waiting for:"
    );

    RCLCPP_INFO(
        logger,
        "  /vision/target_pose"
    );

    RCLCPP_INFO(
        logger,
        "Services:"
    );

    RCLCPP_INFO(
        logger,
        "  /ur7e/plan_to_target"
    );

    RCLCPP_INFO(
        logger,
        "  /ur7e/execute_last_plan"
    );

    RCLCPP_INFO(
        logger,
        "  /ur7e/stop"
    );

    RCLCPP_INFO(
        logger,
        "======================================"
    );


    /*
     * 使用多线程 executor
     */

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(),
        2
    );

    executor.add_node(node);

    executor.spin();


    rclcpp::shutdown();

    return 0;
}