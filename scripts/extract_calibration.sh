#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source "${HOME}/UR7e/config/ur7e.env"

mkdir -p "$(dirname "${CALIBRATION_FILE}")"

echo "正在从 ${ROBOT_IP} 提取 UR7e 标定参数……"
echo "保存位置：${CALIBRATION_FILE}"

ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:="${ROBOT_IP}" \
  target_filename:="${CALIBRATION_FILE}"
