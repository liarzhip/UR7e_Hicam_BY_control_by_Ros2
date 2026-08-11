cmake_minimum_required(VERSION 3.8)

project(ur7e_motion)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall`
    -Wextra
    -Wpedantic
  )
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(
  target_motion_node
  src/target_motion_node.cpp
)

ament_target_dependencies(
  target_motion_node

  rclcpp
  geometry_msgs
  std_srvs
  moveit_ros_planning_interface
)

install(
  TARGETS
  target_motion_node

  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
  config
  launch

  DESTINATION share/${PROJECT_NAME}
)

ament_package()
