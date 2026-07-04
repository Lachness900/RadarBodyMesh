import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Enter Path and Name Here
    package_dir = get_package_share_directory('ti_mmwave_rospkg')
    # path = os.path.join(package_dir,'cfg','1843_Standard.cfg')
    path = os.path.join(package_dir,'cfg','1843_3d.cfg')
    device = "1843"
    name = "/mmWaveCLI"
    command_port = "/dev/ttyACM0"
    command_rate = "115200"
    data_port = "/dev/ttyACM1"
    data_rate = "921600"

    ld = LaunchDescription()

    frame_args = DeclareLaunchArgument(
        name="frame_id",
        default_value="ti_mmwave_0",
        description="frame id to use for publishing topics"
    )
    ConfigParameters = os.path.join(
        package_dir,
        'config',
        'global_params.yaml',
        'launch/*.rviz'
    )
    global_param_node = Node(
        package='ti_mmwave_rospkg',
        executable='ConfigParameterServer',
        name='ConfigParameterServer',
        parameters=[ConfigParameters]
    )

    mmWaveCommSrv = Node(
    package="ti_mmwave_rospkg",
    executable="mmWaveCommSrv",
    name="mmWaveCommSrv",
    output="screen",
    emulate_tty=True,
    parameters=[{
        "command_port": command_port,
        "command_rate": command_rate,
        "data_port": data_port,
        "data_rate": data_rate,
        "max_allowed_elevation_angle_deg": "90",
        "max_allowed_azimuth_angle_deg": "90",
        "frame_id": LaunchConfiguration("frame_id"),
        "mmwavecli_name": name,
        "mmwavecli_cfg": path
        }]
    )

    mmWaveQuickConfig = Node(
        package="ti_mmwave_rospkg",
        executable="mmWaveQuickConfig",
        name="mmWaveQuickConfig",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "mmwavecli_name": name,
            "mmwavecli_cfg": path
            }]
    )

    ParameterParser = Node(
        package="ti_mmwave_rospkg",
        executable="ParameterParser",
        name="ParameterParser",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"device_name": device},
            {"mmwavecli_name": name},
            {"mmwavecli_cfg": path}
        ]
    )

    DataHandlerClass = Node(
        package="ti_mmwave_rospkg",
        executable="DataHandlerClass",
        name="DataHandlerClass",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "mmwavecli_name": name,
            "mmwavecli_cfg": path,
            "command_port": command_port,
            "command_rate": command_rate,
            "data_port": data_port,
            "data_rate": data_rate,
            "max_allowed_elevation_angle_deg": 90,
            "max_allowed_azimuth_angle_deg": 90,
            "frame_id": LaunchConfiguration("frame_id")
        }]
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(package_dir, 'launch', 'rviz.rviz')]
    )

    tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=[
            '--x', '0.0', 
            '--y', '0.0', 
            '--z', '0.0', 
            '--yaw', '0.0', 
            '--pitch', '0.0', 
            '--roll', '0.0', 
            '--frame-id', 'base_link', 
            '--child-frame-id', LaunchConfiguration("frame_id"),
        ],
        output='screen'
    )

    ld.add_action(frame_args)
    ld.add_action(global_param_node)
    ld.add_action(mmWaveCommSrv)
    ld.add_action(mmWaveQuickConfig)
    ld.add_action(ParameterParser)
    ld.add_action(DataHandlerClass)
    ld.add_action(tf_publisher)
    ld.add_action(rviz_node)

    return ld