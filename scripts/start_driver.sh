#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source "${HOME}/UR7e/config/ur7e.env"

if [ ! -f "${CALIBRATION_FILE}" ]; then
    echo "错误：没有找到标定文件："
    echo "${CALIBRATION_FILE}"
    echo
    echo "请先运行："
    echo "${HOME}/UR7e/scripts/extract_calibration.sh"
    exit 1
fi

echo "================================"
echo "启动 UR7e ROS 2 驱动"
echo "机器人型号：${UR_TYPE}"
echo "机器人 IP：${ROBOT_IP}"
echo "标定文件：${CALIBRATION_FILE}"
echo "================================"

exec ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:="${UR_TYPE}" \
  robot_ip:="${ROBOT_IP}" \
  kinematics_params_file:="${CALIBRATION_FILE}" \
  launch_rviz:=false
