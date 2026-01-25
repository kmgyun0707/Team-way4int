# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription, GroupAction 
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time", default="True")
    param_dir = LaunchConfiguration(
        "params_file",
        default=os.path.join(
            get_package_share_directory("carter_navigation"), "params", "carter_navigation_params.yaml"
        ),
    )

    map_dir = LaunchConfiguration(
        "map",
        default=os.path.join(
            get_package_share_directory("carter_navigation"), "maps", "carter_warehouse_navigation.yaml"
        ),
    )

    nav2_bringup_launch_dir = os.path.join(get_package_share_directory("nav2_bringup"), "launch")

    rviz_config_dir = os.path.join(get_package_share_directory("carter_navigation"), "rviz2", "carter_navigation.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value=map_dir, description="Full path to map file to load"),
            DeclareLaunchArgument(
                "params_file", default_value=param_dir, description="Full path to param file to load"
            ),
            DeclareLaunchArgument(
                "use_sim_time", default_value="true", description="Use simulation (Omniverse Isaac Sim) clock if true"
            ),
            
            # 1. RViz 실행
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2_bringup_launch_dir, "rviz_launch.py")),
                launch_arguments={"namespace": "", "use_namespace": "False", "rviz_config": rviz_config_dir}.items(),
            ),
            
            # 2. Nav2 Bringup 실행 (수정됨: GroupAction으로 감싸서 리매핑 적용)
            GroupAction(
                actions=[
                    # Nav2가 '/goal_pose' 토픽을 '/goal_pose_disabled'로 듣게 변경
                    # (결과적으로 RViz의 '/goal_pose' 명령을 무시하게 됨)
                    SetRemap(src='/goal_pose', dst='/goal_pose_disabled'),
                    
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource([nav2_bringup_launch_dir, "/bringup_launch.py"]),
                        launch_arguments={
                            "map": map_dir, 
                            "use_sim_time": use_sim_time, 
                            "params_file": param_dir
                        }.items(),
                    ),
                ]
            ),

            # 3. Pointcloud to LaserScan
            Node(
                package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
                remappings=[('cloud_in', ['/front_3d_lidar/lidar_points']),
                            ('scan', ['/scan'])],
                parameters=[{
                    'target_frame': 'front_3d_lidar',
                    'transform_tolerance': 0.01,
                    'min_height': -0.4,
                    'max_height': 1.5,
                    'angle_min': -1.5708,  
                    'angle_max': 1.5708, 
                    'angle_increment': 0.0087, 
                    'scan_time': 0.3333,
                    'range_min': 0.05,
                    'range_max': 100.0,
                    'use_inf': True,
                    'inf_epsilon': 1.0,
                }],
                name='pointcloud_to_laserscan'
            ),

            # 4. YOLO + Commander 통합 노드 실행
            Node(
                package='carter_navigation',  
                executable='yolo_commander.py', # .py 확장자 필수 (CMakeLists.txt 방식일 때)
                name='yolo_commander',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )