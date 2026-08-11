from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = PathJoinSubstitution(
        [FindPackageShare("ur7e_vision"), "config", "vision.yaml"]
    )

    return LaunchDescription([
        Node(
            package="ur7e_vision",
            executable="hik_camera_node",
            name="hik_camera_node",
            output="screen",
            parameters=[cfg],
        ),
        Node(
            package="ur7e_vision",
            executable="svm_detector_node",
            name="svm_detector_node",
            output="screen",
            parameters=[cfg],
        ),
    ])
