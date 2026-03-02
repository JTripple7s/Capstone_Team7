from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_index_arg = DeclareLaunchArgument('camera_index', default_value='0')
    output_topic_arg = DeclareLaunchArgument('output_topic', default_value='image_raw')
    fps_arg = DeclareLaunchArgument('fps', default_value='20.0')
    backend_arg = DeclareLaunchArgument('capture_backend', default_value='avfoundation')

    camera_node = Node(
        package='app',
        executable='network_camera_node',
        name='network_camera_node',
        output='screen',
        parameters=[{
            'camera_source': 'local',
            'camera_index': LaunchConfiguration('camera_index'),
            'output_topic': LaunchConfiguration('output_topic'),
            'fps': LaunchConfiguration('fps'),
            'capture_backend': LaunchConfiguration('capture_backend'),
        }]
    )

    return LaunchDescription([
        camera_index_arg,
        output_topic_arg,
        fps_arg,
        backend_arg,
        camera_node,
    ])