
cmake_minimum_required(VERSION 3.8)

project(ur7e_motion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ur_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(target_motion_node src/target_motion_node.cpp)
ament_target_dependencies(
  target_motion_node
  rclcpp
  geometry_msgs
  std_srvs
  moveit_ros_planning_interface
)

add_executable(gripper_io_node src/gripper_io_node.cpp)
ament_target_dependencies(
  gripper_io_node
  rclcpp
  std_srvs
  ur_msgs
)

add_executable(pick_task_node src/pick_task_node.cpp)
ament_target_dependencies(
  pick_task_node
  rclcpp
  geometry_msgs
  std_msgs
  std_srvs
)

install(
  TARGETS
  target_motion_node
  gripper_io_node
  pick_task_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
