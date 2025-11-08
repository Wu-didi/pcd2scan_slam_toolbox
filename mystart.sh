ros2 run pcd2scan_slam_toolbox ground_removal_to_scan_ros2


ros2 run pcd2scan_slam_toolbox utm_to_tf 


ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link base_footprint



ros2 run tf2_ros static_transform_publisher 0.5 0 1.2 0 0 0 base_link laser




ros2 launch slam_toolbox online_async_launch.py


ros2 launch slam_toolbox online_async_launch.py slam_params_file:=/home/nvidia/vcii/wudi/pcd2scan_slam_toolbox/pcd2scan_slam_toolbox/myset.yaml

ros2 topic echo /scan

ros2 循环发布bag包
ros2 bag play /home/wudi/slam/hezi_lidar_bag/s9 --loop


save map

ros2 topic hz /rslidar_points_prev



ros2 run nav2_map_server map_saver_cli   -f /home/nvidia/my_map   --ros-args -p save_map_timeout:=10 -p use_sim_time:=true