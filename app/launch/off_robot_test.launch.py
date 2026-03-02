from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Off-robot defaults: camera stream from host machine and safe command remap.
    stream_url_arg = DeclareLaunchArgument(
        'stream_url',
        default_value='http://host.docker.internal:8080/mjpeg'
    )
    image_topic_arg = DeclareLaunchArgument('image_topic', default_value='image_raw')
    camera_fps_arg = DeclareLaunchArgument('camera_fps', default_value='20.0')
    tracking_debug_arg = DeclareLaunchArgument('tracking_debug', default_value='false')
    safe_cmd_vel_topic_arg = DeclareLaunchArgument('safe_cmd_vel_topic', default_value='cmd_vel_test')

    network_camera_node = Node(
        package='app',
        executable='network_camera_node',
        name='network_camera_node',
        output='screen',
        parameters=[{
            'camera_source': 'network',
            'stream_url': LaunchConfiguration('stream_url'),
            'output_topic': LaunchConfiguration('image_topic'),
            'fps': LaunchConfiguration('camera_fps'),
        }],
    )

    tracking_node = Node(
        package='app',
        executable='tracking',
        name='object_tracking',
        output='screen',
        parameters=[{'debug': LaunchConfiguration('tracking_debug')}],
        # Safety: never publish movement commands to real cmd_vel in off-robot tests.
        remappings=[('cmd_vel', LaunchConfiguration('safe_cmd_vel_topic'))],
    )

    return LaunchDescription([
        stream_url_arg,
        image_topic_arg,
        camera_fps_arg,
        tracking_debug_arg,
        safe_cmd_vel_topic_arg,
        network_camera_node,
        tracking_node,
    ])
