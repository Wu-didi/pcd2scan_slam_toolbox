from setuptools import setup, find_packages

package_name = 'pcd2scan_slam_toolbox'

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
                    'utm_to_tf = pcd2scan_slam_toolbox.utm_to_tf:main',
                    'ground_removal_to_scan_ros2 = pcd2scan_slam_toolbox.ground_removal_to_scan_ros2:main',
                    'ground_removal_to_scan = pcd2scan_slam_toolbox.ground_removal_to_scan:main',
                    'simple_ground_removal_to_scan = pcd2scan_slam_toolbox.ground_removal_to_scan_simple:main',
                    'old_simple_ground_removal_to_scan = pcd2scan_slam_toolbox.old_time_ground_removal_to_scan_simple:main',
                    'publish_imu = pcd2scan_slam_toolbox.imu_node:main',
                    'offline_utm_to_tf = pcd2scan_slam_toolbox.offline_utm_to_tf:main',
                    'online_utm_to_tf = pcd2scan_slam_toolbox.online_utm_to_tf:main',
                    'imu_gps_logger = pcd2scan_slam_toolbox.imu_gps_logger:main',
                    'png_map_publisher = pcd2scan_slam_toolbox.png_map_publisher:main',
                    'astar_planner_node = pcd2scan_slam_toolbox.astar_planner_node:main',
                    'local_path_planner_node = pcd2scan_slam_toolbox.local_path_planner_node:main',
                    'lidar_detector = pcd2scan_slam_toolbox.perception.lidar_detector_node:main',
                    'lidar_obstacle_viz = pcd2scan_slam_toolbox.perception.lidar_obstacle_viz_node:main',
                    'lidar_detector_open3d = pcd2scan_slam_toolbox.perception.lidar_detector_open3d_node:main',
                    'fast_lidar_obstacle_detector = pcd2scan_slam_toolbox.perception.fast_lidar_obstacle_detector_node:main',
                    'fast_obstacle_viz = pcd2scan_slam_toolbox.perception.fast_obstacle_viz_node:main',

        ],
    },
)
