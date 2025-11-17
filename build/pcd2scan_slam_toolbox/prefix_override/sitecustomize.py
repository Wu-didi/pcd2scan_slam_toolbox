import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wudi/slam/pcd2scan_slam_toolbox/install/pcd2scan_slam_toolbox'
