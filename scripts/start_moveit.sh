#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source "${HOME}/UR7e/config/ur7e.env"

if [ -f "${HOME}/UR7e/ros2_ws/install/setup.bash" ]; then
    source "${HOME}/UR7e/ros2_ws/install/setup.bash"
fi

echo "启动 UR7e MoveIt 2 和 RViz……"

exec ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:="${UR_TYPE}" \
  launch_rviz:=true
