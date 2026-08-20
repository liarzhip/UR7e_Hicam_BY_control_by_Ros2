from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ur7e_motion",
            namespace="motion",
            executable="gripper_io_node"
        )


    ])