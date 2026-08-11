#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source "${HOME}/UR7e/ros2_ws/install/setup.bash"

if [ -z "${HIK_MVS_PYTHON_PATH:-}" ]; then
    SDK_FILE="$(find /opt/MVS -name MvCameraControl_class.py 2>/dev/null | head -n 1 || true)"
    if [ -z "${SDK_FILE}" ]; then
        echo "ERROR: MvCameraControl_class.py not found under /opt/MVS"
        echo "Install HIKROBOT MVS Linux SDK first."
        exit 1
    fi
    export HIK_MVS_PYTHON_PATH="$(dirname "${SDK_FILE}")"
fi

echo "HIK_MVS_PYTHON_PATH=${HIK_MVS_PYTHON_PATH}"

exec ros2 launch ur7e_vision vision.launch.py
