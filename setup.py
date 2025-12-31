from setuptools import setup, find_packages

package_name = 'Box_AD'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wudi',
    maintainer_email='164662525@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                    'utm_to_tf = Box_AD.utm_to_tf:main',
                    'ground_removal_to_scan_ros2 = Box_AD.ground_removal_to_scan_ros2:main',
                    'ground_removal_to_scan = Box_AD.ground_removal_to_scan:main',
                    'simple_ground_removal_to_scan = Box_AD.ground_removal_to_scan_simple:main',
                    'old_simple_ground_removal_to_scan = Box_AD.old_time_ground_removal_to_scan_simple:main',
                    'publish_imu = Box_AD.imu_node:main',
                    'offline_utm_to_tf = Box_AD.offline_utm_to_tf:main',
                    'online_utm_to_tf = Box_AD.online_utm_to_tf:main',
                    'imu_gps_logger = Box_AD.imu_gps_logger:main',
                    'png_map_publisher = Box_AD.png_map_publisher:main',
                    'astar_planner_node = Box_AD.planner.global_planner_astar_node:main',
                    'local_path_planner_node = Box_AD.planner.local_path_planner_node:main',
                    'lidar_detector = Box_AD.perception.lidar_detector_node:main',
                    'lidar_obstacle_viz = Box_AD.perception.lidar_obstacle_viz_node:main',
                    'lidar_detector_open3d = Box_AD.perception.lidar_detector_open3d_node:main',
                    'fast_lidar_obstacle_detector = Box_AD.perception.fast_lidar_obstacle_detector_node:main',
                    'fast_obstacle_viz = Box_AD.perception.fast_obstacle_viz_node:main',
                    'fast_obstacle_temporal_filter_node = Box_AD.perception.fast_obstacle_temporal_filter_node:main',
                    'path_visualizer = Box_AD.path_visualizer_node:main',
                    'gps_imu_to_utm_pose = Box_AD.gps_imu_to_utm_pose_node:main',
                    'ego_visualizer = Box_AD.ego_visualizer_node:main',
                    'fast_obstacle_rviz_in_map = Box_AD.perception.fast_obstacle_visualizer_node_in_map:main',

        ],
    },
)
