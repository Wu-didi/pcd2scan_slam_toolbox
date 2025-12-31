#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fast_lidar_obstacle_detector_node.py

高效激光障碍物探测节点（无 Open3D、无 DBSCAN）：
- 订阅 /lidar_point_prev (sensor_msgs/PointCloud2)
- 粗略估计全局地面高度
- 在 ROI 内用 2D 栅格统计“高于地面的点”
- 输出每个占用栅格的中心作为障碍物位置

发布：
- /fast_obstacles (std_msgs/String, JSON)
    {
      "stamp": {...},
      "frame_id": "...",
      "has_obstacle": true/false,
      "obstacles": [
        {"x":..., "y":..., "z":..., "count": N},
        ...
      ]
    }
"""

import math
import json
from typing import Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

try:
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    from sensor_msgs import point_cloud2 as pc2


# -------------------- 配置 --------------------

class FastConfig:
    def __init__(self):
        # ROI（沿用你之前的默认）  局部规划不需要很大
        self.xmin = 0.0
        self.xmax = 40.0
        self.ymin = -15.0
        self.ymax = 15.0
        self.zmin = -3.0
        self.zmax = 3.0

        # 估计地面高度用的距离范围
        self.ground_r_min = 3.0   # 忽略靠太近的
        self.ground_r_max = 20.0  # 太远的也忽略

        # 地面高度的分位数（越小越贴近最低点）
        self.ground_percentile = 15.0

        # 判定障碍物的高度条件：z > ground_z + min_obstacle_height
        self.min_obstacle_height = 0.6   # 比地面高 40cm 以上算障碍
        self.max_obstacle_height = 5.0   # 超过这个当作高架/桥，可要可不要

        # 有效探测距离范围
        self.obstacle_r_min = 1.0
        self.obstacle_r_max = 60.0

        # 2D 网格分辨率（米）
        self.grid_resolution = 0.5

        # 每个网格最少点数，低于视为噪声
        self.min_points_per_cell = 3

        # 点云降采样：隔 stride 个点取一个（为 1 则不降采样）
        self.sample_stride = 1  # # 算力够：1；算力紧张：2 或 3


# -------------------- 工具：PointCloud2 -> numpy --------------------

def pc2_to_xyz_array(msg: PointCloud2, stride: int = 1) -> np.ndarray:
    """
    将 PointCloud2 转为 (N, 3) 的 numpy 数组。
    stride>1 时做简单降采样。
    """
    points = []
    for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
        if stride > 1 and (i % stride != 0):
            continue
        points.append((p[0], p[1], p[2]))

    if not points:
        return np.zeros((0, 3), dtype=np.float32)

    arr = np.asarray(points, dtype=np.float32)
    return arr


# -------------------- 主检测逻辑（纯 numpy） --------------------

def fast_detect_obstacles(xyz: np.ndarray, cfg: FastConfig):
    """
    输入：xyz (N,3)
    输出：列表 obstacles = [{"x":..,"y":..,"z":..,"count":..}, ...]
    """
    if xyz.shape[0] == 0:
        return []

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # 1) ROI 裁剪
    roi_mask = (
        (x >= cfg.xmin) & (x <= cfg.xmax) &
        (y >= cfg.ymin) & (y <= cfg.ymax) &
        (z >= cfg.zmin) & (z <= cfg.zmax)
    )
    if not np.any(roi_mask):
        return []

    x = x[roi_mask]
    y = y[roi_mask]
    z = z[roi_mask]

    # 2) 粗略估计地面高度（用近处一圈的低分位数）
    r = np.sqrt(x * x + y * y)
    ground_mask = (
        (r >= cfg.ground_r_min) &
        (r <= cfg.ground_r_max)
    )
    if np.any(ground_mask):
        ground_z = np.percentile(z[ground_mask], cfg.ground_percentile)
    else:
        # 兜底：直接取整个 ROI 里 z 的低分位
        ground_z = np.percentile(z, cfg.ground_percentile)

    # 3) 挑出疑似障碍物点
    obstacle_mask = (
        (r >= cfg.obstacle_r_min) &
        (r <= cfg.obstacle_r_max) &
        (z >= ground_z + cfg.min_obstacle_height) &
        (z <= ground_z + cfg.max_obstacle_height)
    )

    if not np.any(obstacle_mask):
        return []

    xo = x[obstacle_mask]
    yo = y[obstacle_mask]
    zo = z[obstacle_mask]

    # 4) 投到 2D 网格，按 (ix, iy) 统计
    res = cfg.grid_resolution
    ix = np.floor((xo - cfg.xmin) / res).astype(int)
    iy = np.floor((yo - cfg.ymin) / res).astype(int)

    cells = {}  # (ix, iy) -> [sum_x, sum_y, sum_z, count]
    for xi, yi, zi, gx, gy in zip(xo, yo, zo, ix, iy):
        key = (int(gx), int(gy))
        if key not in cells:
            cells[key] = [0.0, 0.0, 0.0, 0]
        acc = cells[key]
        acc[0] += float(xi)
        acc[1] += float(yi)
        acc[2] += float(zi)
        acc[3] += 1

    obstacles = []
    for (gx, gy), (sx, sy, sz, cnt) in cells.items():
        if cnt < cfg.min_points_per_cell:
            continue
        cx = sx / cnt
        cy = sy / cnt
        cz = sz / cnt
        obstacles.append({
            "x": float(cx),
            "y": float(cy),
            "z": float(cz),
            "count": int(cnt)
        })

    return obstacles


# -------------------- ROS2 节点 --------------------

class FastLidarObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__("fast_lidar_obstacle_detector")

        self.cfg = FastConfig()

        self.sub = self.create_subscription(
            PointCloud2,
            "/rslidar_points_prev",
            self.lidar_callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            "/fast_obstacles",
            10
        )

        self.get_logger().info(
            "FastLidarObstacleDetectorNode started. "
            "Subscribing /lidar_point_prev, publishing /fast_obstacles"
        )

    def lidar_callback(self, msg: PointCloud2):
        # PointCloud2 -> numpy
        xyz = pc2_to_xyz_array(msg, stride=self.cfg.sample_stride)
        if xyz.shape[0] == 0:
            return

        # Fast obstacle detection
        obstacles = fast_detect_obstacles(xyz, self.cfg)

        result = {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frame_id": msg.header.frame_id,
            "has_obstacle": bool(len(obstacles) > 0),
            "obstacles": obstacles
        }

        out = String()
        out.data = json.dumps(result)
        self.pub.publish(out)

        # 打一点日志方便调试（别太频繁）
        self.get_logger().debug(f"detected {len(obstacles)} cells")


def main(args=None):
    rclpy.init(args=args)
    node = FastLidarObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
