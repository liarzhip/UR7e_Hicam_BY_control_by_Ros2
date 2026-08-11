#!/usr/bin/env bash
set -e

source "${HOME}/UR7e/config/ur7e.env"

echo "================================"
echo "UR7e 网络检查"
echo "机器人 IP：${ROBOT_IP}"
echo "电脑 IP：  ${PC_IP}"
echo "================================"

echo
echo "[1] 检查电脑 IP"
if ip -4 addr | grep -q "${PC_IP}"; then
    echo "成功：Ubuntu 已配置 ${PC_IP}"
else
    echo "失败：没有找到 ${PC_IP}"
    echo "请检查有线网卡的 IPv4 设置。"
    exit 1
fi

echo
echo "[2] 检查到机器人的路由"
ip route get "${ROBOT_IP}"

echo
echo "[3] Ping UR7e"
ping -c 4 "${ROBOT_IP}"

echo
echo "网络通信正常。"

