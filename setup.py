from setuptools import setup

package_name = 'pcd2scan_slam_toolbox'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
                    'simple_ground_removal_to_scan = pcd2scan_slam_toolbox.ground_removal_to_scan_simple:main'
        ],
    },
)
