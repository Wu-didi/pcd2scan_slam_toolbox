# PCD2Scan SLAM Toolbox

一个用于 ROS2 的 SLAM 工具包，提供点云处理、传感器数据采集、坐标转换、地图发布和路径规划等功能。

## 📋 目录

- [功能概述](#功能概述)
- [系统要求](#系统要求)
- [安装与编译](#安装与编译)
- [节点说明](#节点说明)
- [使用示例](#使用示例)
- [配置参数](#配置参数)
- [话题与服务](#话题与服务)
- [常见问题](#常见问题)

## 🎯 功能概述

本工具包主要包含以下功能模块：

1. **点云处理**：将点云文件（PCD/PLY/BIN/NPY等）转换为 ROS2 LaserScan 消息
2. **传感器驱动**：CAN 总线 IMU/GPS 数据采集与发布
3. **坐标转换**：GPS/UTM 坐标到 TF 变换（支持在线和离线模式）
4. **地图发布**：PNG 图像转换为 OccupancyGrid 地图
5. **路径规划**：A* 全局路径规划 + DWA/Lattice 混合局部控制
6. **数据记录**：IMU/GPS 数据 CSV 日志记录

## 🔧 系统要求

- **ROS2 版本**：Humble 或更高版本
- **Python 版本**：Python 3.8+
- **依赖库**：
  - `numpy`
  - `opencv-python` (cv2)
  - `open3d`
  - `pyproj`
  - `python-can`
  - `rclpy`
  - `sensor_msgs`
  - `nav_msgs`
  - `geometry_msgs`
  - `tf2_ros`

## 📦 安装与编译

### 1. 克隆仓库

```bash
cd ~/your_workspace/src
git clone <repository_url> pcd2scan_slam_toolbox
```

### 2. 安装 Python 依赖

```bash
pip3 install numpy opencv-python open3d pyproj python-can
```

### 3. 编译工作空间

```bash
cd ~/your_workspace
colcon build --packages-select pcd2scan_slam_toolbox
source install/setup.bash
```

## 🚀 节点说明

### 1. 点云转激光扫描节点

#### `ground_removal_to_scan_ros2`

将点云文件转换为激光扫描数据，执行地面去除并发布为 `LaserScan` 消息。

**功能特性**：
- 支持多种点云格式：PCD, PLY, XYZ, BIN, NPY 等
- Ray-Ground 地面去除算法
- 2D 投影并按角度排序
- 可处理单个文件或目录批量处理

**使用方法**：
```bash
ros2 run pcd2scan_slam_toolbox ground_removal_to_scan_ros2 \
  --ros-args \
  -p input_path:=/path/to/pointcloud.pcd \
  -p frame_id:=laser \
  -p publish_rate:=10.0
```

**发布话题**：
- `/scan` (sensor_msgs/LaserScan)：处理后的激光扫描数据
- `/raw_cloud` (sensor_msgs/PointCloud2)：原始点云（可选）
- `/nonground_cloud` (sensor_msgs/PointCloud2)：去地面后的点云（可选）

**参数**：
- `input_path`：点云文件路径或目录
- `frame_id`：坐标系名称（默认：`laser`）
- `publish_rate`：发布频率（Hz，默认：10.0）
- `n_sectors`：角度分段数（默认：180）
- `min_range`：最小检测距离（m，默认：1.5）
- `max_range`：最大检测距离（m，默认：80.0）
- `max_slope_deg`：最大地面坡度（度，默认：18.0）

#### 其他点云处理节点

- `ground_removal_to_scan`：基础版本
- `simple_ground_removal_to_scan`：简化版本
- `old_simple_ground_removal_to_scan`：旧版简化版本

### 2. IMU/GPS 传感器节点

#### `publish_imu` (imu_node)

从 CAN 总线读取 IMU 和 GPS 数据，并发布为 ROS2 消息。

**功能特性**：
- CAN 总线接口（socketcan）
- 实时解析 IMU 加速度、角速度、姿态角
- GPS 经纬度、海拔高度解析
- 50Hz IMU 数据发布

**使用方法**：
```bash
# 确保 CAN 接口已配置
sudo ip link set can0 up type can bitrate 500000

ros2 run pcd2scan_slam_toolbox publish_imu
```

**发布话题**：
- `/IMU` (sensor_msgs/Imu)：IMU 数据（50Hz）
- `/gps/fix` (sensor_msgs/NavSatFix)：GPS 定位数据

**CAN 消息 ID**：
- `0x500`：加速度数据
- `0x501`：角速度数据
- `0x502`：姿态角数据
- `0x503`：GPS 海拔数据
- `0x504`：GPS 经纬度数据

#### `imu_gps_logger`

记录 IMU 和 GPS 数据到 CSV 文件。

**使用方法**：
```bash
ros2 run pcd2scan_slam_toolbox imu_gps_logger
```

**输出文件**：
- `logs/imu.csv`：IMU 数据（时间戳、姿态、角速度、加速度）
- `logs/gps.csv`：GPS 数据（时间戳、经纬度、海拔）

**订阅话题**：
- `/IMU` (sensor_msgs/Imu)
- `/gps/fix` (sensor_msgs/NavSatFix)

### 3. 坐标转换节点

#### `online_utm_to_tf`

在线模式：实时将 GPS/UTM 坐标转换为 TF 变换。

**功能特性**：
- 经纬度转 UTM 坐标
- 结合 IMU 姿态角计算 TF
- 实时发布 `odom -> base_link` 变换

**使用方法**：
```bash
ros2 run pcd2scan_slam_toolbox online_utm_to_tf \
  --ros-args \
  -p utm_zone_number:=49 \
  -p offset_x:=1240249.0084191752 \
  -p offset_y:=3555312.0655697277
```

**订阅话题**：
- `/IMU` (sensor_msgs/Imu)
- `/gps/fix` (sensor_msgs/NavSatFix)

**发布 TF**：
- `odom -> base_link`

#### `offline_utm_to_tf`

离线模式：从 CSV 文件读取 GPS/IMU 数据并生成 TF。

**使用方法**：
```bash
ros2 run pcd2scan_slam_toolbox offline_utm_to_tf \
  --ros-args \
  -p gps_csv:=logs/gps.csv \
  -p imu_csv:=logs/imu.csv
```

#### `utm_to_tf`

基础版本 UTM 到 TF 转换节点。

### 4. 地图发布节点

#### `png_map_publisher`

将 PNG 图像文件转换为 OccupancyGrid 地图并发布。

**功能特性**：
- 支持 PNG 灰度图像
- 自动翻转坐标系（图像坐标转地图坐标）
- 可配置分辨率、原点、坐标系

**使用方法**：
```bash
ros2 run pcd2scan_slam_toolbox png_map_publisher \
  --ros-args \
  -p image_path:=/path/to/map.png \
  -p resolution:=0.1 \
  -p origin_x:=-50.0 \
  -p origin_y:=-80.0 \
  -p frame_id:=map
```

**发布话题**：
- `/map` (nav_msgs/OccupancyGrid)：栅格地图（1Hz）

**参数**：
- `image_path`：PNG 图像文件路径
- `resolution`：地图分辨率（m/pixel，默认：0.1）
- `origin_x`：地图原点 X 坐标（m，默认：0.0）
- `origin_y`：地图原点 Y 坐标（m，默认：0.0）
- `frame_id`：坐标系名称（默认：`map`）

**图像格式**：
- 白色像素（>250）：自由空间（0）
- 黑色像素（<5）：障碍物（100）
- 其他：未知区域（-1）

### 5. 路径规划节点

#### `astar_planner_node`

基于 A* 算法的全局路径规划节点。

**功能特性**：
- A* 搜索算法（8 邻域）
- 障碍物膨胀（考虑机器人尺寸）
- Line-of-Sight 路径简化
- 车辆动力学约束平滑
- 曲率约束（最小转弯半径）

**使用方法**：
```bash
ros2 run pcd2scan_slam_toolbox astar_planner_node \
  --ros-args \
  -p robot_radius:=0.5 \
  -p min_turning_radius:=2.0 \
  -p smooth_iterations:=150
```

**订阅话题**：
- `/map` (nav_msgs/OccupancyGrid)：静态地图
- `/initialpose` (geometry_msgs/PoseWithCovarianceStamped)：起点
- `/goal_pose` (geometry_msgs/PoseStamped)：终点

**发布话题**：
- `/global_path` (nav_msgs/Path)：全局路径

**参数**：
- `robot_radius`：机器人半径（m，用于障碍物膨胀，默认：0.5）
- `min_turning_radius`：最小转弯半径（m，默认：2.0）
- `smooth_iterations`：平滑迭代次数（默认：150）
- `smooth_weight_data`：数据项权重（默认：0.15）
- `smooth_weight_smooth`：平滑项权重（默认：0.45）
- `curvature_gain`：曲率增益（默认：0.4）
- `path_sample_step`：路径采样步长（m，默认：0.2）

**算法流程**：
1. 接收地图、起点、终点
2. 障碍物膨胀（基于 `robot_radius`）
3. A* 搜索路径
4. Line-of-Sight 简化
5. 车辆动力学平滑
6. 路径重采样
7. 计算朝向角并发布

#### `local_path_planner_node`

混合 DWA/Lattice 局部规划与驱动节点，基于全局路径、激光雷达和机器人速度约束实时生成安全的轨迹与 `cmd_vel`。

**功能特性**：
- Dynamic Window Approach 采样并严格满足线速度/角速度/加速度约束
- Lattice 偏置，引导采样沿离散转向分支贴合全局路径
- 基于 LaserScan 预测动态障碍（通过相邻帧的距离变化预估未来位置）
- 对候选轨迹评估航向误差、路径偏差、速度收益、清距和碰撞时间
- 同时发布最优局部轨迹 (`/local_path`)、避障后更新的路径 (`/update_global_path`) 与驱动指令 (`/cmd_vel`)

**使用方法**：
```bash
ros2 run pcd2scan_slam_toolbox local_path_planner_node \
  --ros-args \
  -p global_path_topic:=/global_path \
  -p pose_topic:=/amcl_pose \
  -p pose_type:=PoseWithCovarianceStamped \
  -p velocity_topic:=/odom \
  -p laser_topic:=/scan \
  -p cmd_vel_topic:=/cmd_vel \
  -p update_global_path_topic:=/update_global_path \
  -p lookahead_distance:=6.0 \
  -p robot_radius:=0.7 \
  -p max_speed:=1.5 \
  -p max_acceleration:=1.0 \
  -p dynamic_obstacle_horizon:=1.0
```

**订阅话题**：
- `/global_path` (nav_msgs/Path)：全局路径轨迹
- 定位话题（PoseStamped / PoseWithCovarianceStamped / Odometry）：机器人姿态
- `/odom` (nav_msgs/Odometry，可选)：速度输入（若 `pose_type` 为 Odometry 可复用同一话题）
- `/scan` (sensor_msgs/LaserScan)：动态障碍观测

**发布话题**：
- `/local_path` (nav_msgs/Path)：当前最优局部轨迹
- `/update_global_path` (nav_msgs/Path)：将局部避障结果拼接回全局的更新路径
- `/cmd_vel` (geometry_msgs/Twist)：驱动控制命令

**主要参数**：
- `global_path_topic`：全局路径话题名（默认 `global_path`）
- `local_path_topic`：局部轨迹话题名（默认 `local_path`）
- `update_global_path_topic`：避障后更新全局路径话题名（默认 `update_global_path`）
- `cmd_vel_topic`：控制输出话题名（默认 `cmd_vel`）
- `pose_topic` / `pose_type`：定位来源及类型（默认 `amcl_pose` / `PoseWithCovarianceStamped`）
- `velocity_topic`：速度来源（默认 `odom`）
- `laser_topic`：激光雷达话题（默认 `scan`，要求与 base_link 固定变换）
- `lookahead_distance`：追踪前瞻距离（m）
- `goal_tolerance`：终点容差（m）
- `robot_radius`：安全半径（m，用于碰撞检测）
- `control_frequency`：控制循环频率（Hz）
- `predict_time` / `predict_dt`：仿真时域与步长
- `velocity_samples` / `yaw_samples`：动态窗口采样数量
- `max_speed` / `min_speed`：线速度上下限
- `max_acceleration` / `max_deceleration`：线速度加/减速度约束
- `max_yaw_rate` / `max_yaw_acceleration`：角速度约束
- `path_follow_weight`, `heading_weight`, `velocity_weight`, `clearance_weight`, `ttc_weight`：代价函数权重
- `obstacle_max_range`：激光障碍最大取值距离
- `dynamic_obstacle_horizon`：动态障碍预测时间（s）
- `lattice_yaw_bias`：Lattice 偏置列表，为 yaw 采样添加额外候选

## 📖 使用示例

### 完整 SLAM 流程示例

```bash
# 终端1：发布 PNG 地图
ros2 run pcd2scan_slam_toolbox png_map_publisher \
  --ros-args \
  -p image_path:=/path/to/map.png \
  -p resolution:=0.1

# 终端2：点云转激光扫描
ros2 run pcd2scan_slam_toolbox ground_removal_to_scan_ros2 \
  --ros-args \
  -p input_path:=/path/to/pointclouds/

# 终端3：发布 TF 变换（如果需要）
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link base_footprint
ros2 run tf2_ros static_transform_publisher 0.5 0 1.2 0 0 0 base_link laser

# 终端4：启动 SLAM
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=/path/to/config/mapper_params_online_async.yaml

# 终端5：A* 路径规划
ros2 run pcd2scan_slam_toolbox astar_planner_node

# 终端6：局部路径规划
ros2 run pcd2scan_slam_toolbox local_path_planner_node
```

### 传感器数据采集示例

```bash
# 终端1：启动 CAN IMU 节点
ros2 run pcd2scan_slam_toolbox publish_imu

# 终端2：记录数据
ros2 run pcd2scan_slam_toolbox imu_gps_logger

# 终端3：UTM 坐标转换
ros2 run pcd2scan_slam_toolbox online_utm_to_tf
```

### 在 RViz 中可视化

```bash
# 启动 RViz
rviz2

# 添加显示项：
# - Map: /map
# - Path: /global_path, /local_path
# - LaserScan: /scan
# - TF: 显示坐标系树
```

## ⚙️ 配置参数

### SLAM 配置文件

配置文件位于 `config/` 目录：

- `mapper_params_online_async.yaml`：在线异步 SLAM 参数
- `myset.yaml`：自定义 SLAM 参数

主要参数说明：
- `mode`：`mapping`（建图）或 `localization`（定位）
- `resolution`：地图分辨率（m/pixel）
- `map_update_interval`：地图更新间隔（s）
- `minimum_travel_distance`：最小移动距离（m）
- `do_loop_closing`：是否启用回环检测

## 📡 话题与服务

### 主要话题列表

| 话题名称 | 消息类型 | 说明 |
|---------|---------|------|
| `/scan` | sensor_msgs/LaserScan | 激光扫描数据 |
| `/map` | nav_msgs/OccupancyGrid | 栅格地图 |
| `/IMU` | sensor_msgs/Imu | IMU 数据 |
| `/gps/fix` | sensor_msgs/NavSatFix | GPS 定位数据 |
| `/global_path` | nav_msgs/Path | 全局路径 |
| `/local_path` | nav_msgs/Path | 局部路径/最优局部轨迹 |
| `/update_global_path` | nav_msgs/Path | 避障后更新的全局路径 |
| `/cmd_vel` | geometry_msgs/Twist | 局部规划输出速度 |
| `/initialpose` | geometry_msgs/PoseWithCovarianceStamped | 初始位姿（起点） |
| `/goal_pose` | geometry_msgs/PoseStamped | 目标位姿（终点） |

### TF 变换树

典型的 TF 树结构：
```
map
 └── odom
      └── base_link
           └── base_footprint
           └── laser
```

## ❓ 常见问题

### 1. 点云文件无法读取

**问题**：`ground_removal_to_scan_ros2` 报错无法读取点云文件

**解决**：
- 检查文件路径是否正确
- 确认文件格式是否支持（PCD, PLY, BIN, NPY 等）
- 检查文件权限

### 2. CAN 总线无法连接

**问题**：`publish_imu` 无法读取 CAN 数据

**解决**：
```bash
# 检查 CAN 接口状态
ip link show can0

# 配置 CAN 接口
sudo ip link set can0 up type can bitrate 500000

# 检查权限
sudo usermod -a -G dialout $USER
# 重新登录后生效
```

### 3. 路径规划失败

**问题**：`astar_planner_node` 无法找到路径

**解决**：
- 检查起点和终点是否在自由空间内
- 检查 `robot_radius` 参数是否过大
- 确认地图已正确发布到 `/map` 话题
- 在 RViz 中检查地图和起终点位置

### 4. TF 变换缺失

**问题**：坐标转换节点无法工作

**解决**：
- 使用 `ros2 run tf2_ros tf2_echo map odom` 检查 TF 树
- 确认所有必要的静态 TF 已发布
- 检查坐标系名称是否匹配

### 5. 地图显示异常

**问题**：PNG 地图在 RViz 中显示不正确

**解决**：
- 检查 `origin_x` 和 `origin_y` 参数
- 确认图像格式（灰度图，白色=自由，黑色=障碍）
- 检查 `resolution` 参数是否与实际地图匹配

## 📝 开发说明

### 代码结构

```
pcd2scan_slam_toolbox/
├── pcd2scan_slam_toolbox/          # Python 包
│   ├── astar_planner_node.py       # A* 路径规划
│   ├── local_path_planner_node.py   # 局部路径规划
│   ├── ground_removal_to_scan_ros2.py  # 点云处理
│   ├── imu_node.py                 # CAN IMU 驱动
│   ├── imu_gps_logger.py           # 数据记录
│   ├── online_utm_to_tf.py         # 在线坐标转换
│   ├── offline_utm_to_tf.py        # 离线坐标转换
│   └── png_map_publisher.py        # 地图发布
├── config/                          # 配置文件
│   ├── mapper_params_online_async.yaml
│   └── myset.yaml
├── logs/                            # 数据日志目录
├── package.xml                      # ROS2 包定义
├── setup.py                         # Python 安装配置
└── README.md                        # 本文档
```

### 添加新节点

1. 在 `pcd2scan_slam_toolbox/` 目录创建新的 Python 文件
2. 实现 ROS2 Node 类
3. 在 `setup.py` 的 `entry_points` 中添加节点入口
4. 重新编译：`colcon build --packages-select pcd2scan_slam_toolbox`

## 📄 许可证

TODO: License declaration

## 👤 维护者

- **wudi** - 164662525@qq.com

## 🙏 致谢

本工具包基于以下开源项目：
- ROS2
- Open3D
- OpenCV
- SLAM Toolbox

---

**注意**：本工具包仍在开发中，部分功能可能不稳定。如有问题请提交 Issue。
