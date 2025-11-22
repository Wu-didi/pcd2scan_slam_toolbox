#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
urban_lidar_detector_node.py

基于原来的 urban_lidar_autoware_like.py 改写为 ROS2 节点版本：
- 订阅 /lidar_point_prev (sensor_msgs/PointCloud2)
- 进行 ROI 裁剪 + 体素降采样 + Ray-Ground Filter + 自适应 DBSCAN 聚类 + OBB 拟合 + 启发式分类
- 将检测到的障碍物结果以 JSON 字符串形式发布到 /lidar_obstacles (std_msgs/String)

JSON 格式示例：
{
  "stamp": {"sec": 123, "nanosec": 456},
  "frame_id": "lidar",
  "objects": [
    {
      "id": 0,
      "label": "vehicle",
      "center": [x, y, z],
      "size": [L, W, H],
      "yaw": yaw
    },
    ...
  ]
}
"""

import math
import json
from typing import List, Tuple, Dict

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


# ============================ 工具函数 ============================

def crop_roi(pc: o3d.geometry.PointCloud,
             x_lim: Tuple[float, float],
             y_lim: Tuple[float, float],
             z_lim: Tuple[float, float]) -> o3d.geometry.PointCloud:
    if pc.is_empty():
        return pc
    pts = np.asarray(pc.points)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    m = (x >= x_lim[0]) & (x <= x_lim[1]) & \
        (y >= y_lim[0]) & (y <= y_lim[1]) & \
        (z >= z_lim[0]) & (z <= z_lim[1])
    idx = np.where(m)[0]
    return pc.select_by_index(idx)


def radius_outlier(pc: o3d.geometry.PointCloud, nb_points=8, radius=0.45):
    if pc.is_empty():
        return pc
    idx = pc.remove_radius_outlier(nb_points=nb_points, radius=radius)[1]
    return pc.select_by_index(idx)


def atan2_yaw_from_R(Rm: np.ndarray) -> float:
    return float(math.atan2(Rm[1, 0], Rm[0, 0]))


def reorder_lw(sx, sy):
    # 使 length >= width（便于筛选）
    if sx >= sy:
        return sx, sy
    else:
        return sy, sx


# ======================== Ray-Ground Filter =======================

def ray_ground_filter(pc: o3d.geometry.PointCloud,
                      n_sectors: int = 180,
                      min_range: float = 1.5,
                      max_range: float = 80.0,
                      max_slope_deg: float = 18.0,
                      seed_dist: float = 3.0,
                      height_thresh: float = 0.2,
                      seed_percentile: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    与 Autoware 思路一致的射线/扇区地面滤除：
    - 将点按方位角划分为扇区，按距离升序遍历
    - 就近若干米（seed_dist）取低分位数作为初始地面高度
    - 若当前点 z <= ground_z + tan(max_slope)*Δr + height_thresh -> 地面；否则非地面
    返回：ground_mask, nonground_mask（bool）
    """
    if pc.is_empty():
        n = 0
        return np.zeros((n,), dtype=bool), np.zeros((n,), dtype=bool)

    pts = np.asarray(pc.points)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    # [0, n_sectors-1]
    sector = ((theta + np.pi) / (2 * np.pi) * n_sectors).astype(int)
    sector = np.clip(sector, 0, n_sectors - 1)

    # 仅用合理距离内点
    valid = (r >= min_range) & (r <= max_range)
    ground = np.zeros(len(pts), dtype=bool)
    non_ground = np.zeros(len(pts), dtype=bool)
    slope_tan = math.tan(math.radians(max_slope_deg))

    for s in range(n_sectors):
        idx = np.where(valid & (sector == s))[0]
        if idx.size == 0:
            continue
        # 按距离排序
        si = idx[np.argsort(r[idx])]
        rs = r[si]
        zs = z[si]

        # 初始地面高度：近处若干米的低分位数
        near_mask = rs - rs[0] <= seed_dist
        if np.any(near_mask):
            seed_z = np.percentile(zs[near_mask], seed_percentile)
        else:
            seed_z = float(np.min(zs))
        ground_z = seed_z
        prev_r = rs[0]

        for j, pj in enumerate(si):
            dr = rs[j] - prev_r
            thresh = slope_tan * dr + height_thresh
            if z[pj] <= ground_z + thresh:
                ground[pj] = True
                # 缓慢更新地面高度，抑制抖动
                ground_z = 0.9 * ground_z + 0.1 * z[pj]
                prev_r = rs[j]
            else:
                non_ground[pj] = True
                prev_r = rs[j]

    # 兜底处理
    undecided = ~(ground | non_ground)
    if np.any(undecided):
        g2 = z < (np.percentile(z[valid], 10.0) + height_thresh)
        ground[undecided & g2] = True
        non_ground[undecided & (~g2)] = True

    return ground, non_ground


# ======================= 自适应聚类（DBSCAN） =======================

def adaptive_dbscan(pc: o3d.geometry.PointCloud,
                    r_bins: List[Tuple[float, float]],
                    eps_list: List[float],
                    min_points: int = 12) -> List[np.ndarray]:
    """
    将点按距离分段，每段使用不同 eps 做 DBSCAN，合并簇。
    """
    if pc.is_empty():
        return []
    pts = np.asarray(pc.points)
    d = np.linalg.norm(pts[:, :2], axis=1)

    clusters = []
    for (l, r), eps in zip(r_bins, eps_list):
        sel = np.where((d >= l) & (d < r))[0]
        if sel.size == 0:
            continue
        sub = pc.select_by_index(sel)
        labels = np.array(sub.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            continue
        for k in range(labels.max() + 1):
            sub_idx = sel[np.where(labels == k)[0]]
            if sub_idx.size >= min_points:
                clusters.append(sub_idx)
    return clusters


# ========================== 目标构型 & 分类 ==========================

def obb_xy_and_info(cluster: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.OrientedBoundingBox, Dict]:
    obb = cluster.get_oriented_bounding_box()
    yaw = atan2_yaw_from_R(np.asarray(obb.R))
    sx, sy, sz = obb.extent
    L, W = reorder_lw(sx, sy)
    info = {
        "center": np.asarray(obb.center).tolist(),
        "size": [float(L), float(W), float(sz)],
        "yaw": float(yaw)
    }
    return obb, info


def heuristic_classify(L, W, H) -> str:
    # 极简启发式：仅用于快速筛查
    # vehicle
    if (2.0 <= L <= 6.0) and (1.2 <= W <= 2.8) and (1.2 <= H <= 3.2):
        return "vehicle"
    # cyclist（含摩托/自行车+人）
    if (1.0 <= L <= 2.5) and (0.4 <= W <= 1.2) and (1.2 <= H <= 2.2):
        return "cyclist"
    # pedestrian
    if (0.2 <= L <= 1.2) and (0.2 <= W <= 1.0) and (1.2 <= H <= 2.2):
        return "pedestrian"
    return "unknown"


# ============================== 配置 ==============================

class DetectorConfig:
    """等价于原来 argparse 的默认参数配置"""
    def __init__(self):
        # ROI
        self.xmin = 0.0
        self.xmax = 70.0
        self.ymin = -15.0
        self.ymax = 15.0
        self.zmin = -3.0
        self.zmax = 3.0

        # 预处理
        self.voxel = 0.08
        self.ro_nb = 8
        self.ro_radius = 0.45

        # Ray-Ground
        self.sectors = 180
        self.min_range = 1.5
        self.max_range = 80.0
        self.max_slope_deg = 18.0
        self.seed_dist = 3.0
        self.height_thresh = 0.2
        self.seed_pct = 10.0

        # 自适应聚类
        self.mid_dist = 25.0
        self.far_dist = 45.0
        self.eps_near = 0.35
        self.eps_mid = 0.6
        self.eps_far = 0.9
        self.min_pts = 12

        # 尺寸/距离约束
        self.min_det_dist = 1.0
        self.max_det_dist = 70.0
        self.min_L = 0.3
        self.max_L = 7.0
        self.min_W = 0.2
        self.max_W = 3.5
        self.min_H = 0.2
        self.max_H = 3.5

        # 其他
        self.x_offset = 0.0  # 如果需要在 x 上修正一个偏移，可以用


# ============================== 检测主流程 ==============================

def detect_objects_in_cloud(pc_in: o3d.geometry.PointCloud,
                            cfg: DetectorConfig) -> List[Dict]:
    """
    对单帧点云进行处理，返回每个障碍物的 info 列表。
    info 字段：
        - id: int
        - label: str
        - center: [x, y, z]
        - size: [L, W, H]
        - yaw: float
    """

    # 0) ROI 裁剪
    pc = crop_roi(
        pc_in,
        (cfg.xmin, cfg.xmax),
        (cfg.ymin, cfg.ymax),
        (cfg.zmin, cfg.zmax)
    )

    # 1) 体素降采样
    if cfg.voxel > 1e-6:
        pc = pc.voxel_down_sample(cfg.voxel)

    # 2) 半径离群
    if cfg.ro_nb > 0:
        pc = radius_outlier(pc, nb_points=cfg.ro_nb, radius=cfg.ro_radius)

    if pc.is_empty():
        return []

    # 3) Ray-Ground Filter
    gmask, nmask = ray_ground_filter(
        pc,
        n_sectors=cfg.sectors,
        min_range=cfg.min_range,
        max_range=cfg.max_range,
        max_slope_deg=cfg.max_slope_deg,
        seed_dist=cfg.seed_dist,
        height_thresh=cfg.height_thresh,
        seed_percentile=cfg.seed_pct
    )

    # 前景：非地面点
    fg = pc.select_by_index(np.where(nmask)[0])
    if fg.is_empty():
        return []

    # 4) 距离自适应聚类
    bins = [(0, cfg.mid_dist), (cfg.mid_dist, cfg.far_dist), (cfg.far_dist, 1e6)]
    epses = [cfg.eps_near, cfg.eps_mid, cfg.eps_far]
    clusters_idx = adaptive_dbscan(fg, bins, epses, min_points=cfg.min_pts)

    infos = []

    for idx in clusters_idx:
        sub = fg.select_by_index(idx)
        obb, inf = obb_xy_and_info(sub)
        L, W, H = inf["size"]

        # 距离过滤
        center = np.asarray(obb.center)
        dist = float(np.linalg.norm(center[:2]))
        if not (cfg.min_det_dist <= dist <= cfg.max_det_dist):
            continue

        # 尺寸过滤
        if not (cfg.min_L <= L <= cfg.max_L and
                cfg.min_W <= W <= cfg.max_W and
                cfg.min_H <= H <= cfg.max_H):
            continue

        # 启发式类别
        label = heuristic_classify(L, W, H)
        inf.update({
            "id": len(infos),
            "label": label
        })

        # 如果需要去掉 x 偏移，可以在这里减去
        inf["center"][0] = float(inf["center"][0] - cfg.x_offset)

        infos.append(inf)

    return infos


# ============================ PointCloud2 转换 ============================

def pointcloud2_to_o3d(pc_msg: PointCloud2) -> o3d.geometry.PointCloud:
    """
    将 ROS2 的 PointCloud2 转为 Open3D 的 PointCloud（只取 x,y,z）
    """
    # 读取所有点
    points = []
    for p in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

    pc = o3d.geometry.PointCloud()
    if len(points) > 0:
        pts_np = np.asarray(points, dtype=np.float64)
        pc.points = o3d.utility.Vector3dVector(pts_np)
    return pc


# ============================== ROS2 节点 ==============================

class UrbanLidarDetectorNode(Node):
    def __init__(self):
        super().__init__("urban_lidar_detector")

        # 参数可以以后改为 declare_parameter，这里直接使用默认配置
        self.cfg = DetectorConfig()

        # 订阅 LiDAR 点云
        self.subscription = self.create_subscription(
            PointCloud2,
            "/rslidar_points_prev",
            self.lidar_callback,
            10
        )

        # 发布障碍物检测结果
        self.pub_obstacles = self.create_publisher(
            String,
            "/lidar_obstacles",
            10
        )

        self.get_logger().info("Urban LiDAR Detector Node started. Listening on /rslidar_points_prev")

    def lidar_callback(self, msg: PointCloud2):
        try:
            # 1) PointCloud2 -> Open3D
            pc_o3d = pointcloud2_to_o3d(msg)
            if pc_o3d.is_empty():
                self.get_logger().warn("Received empty pointcloud, skip.")
                return

            # 2) 目标检测
            infos = detect_objects_in_cloud(pc_o3d, self.cfg)

            # 3) 打包为 JSON，发布
            result = {
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec)
                },
                "frame_id": msg.header.frame_id,
                "objects": infos
            }

            msg_out = String()
            msg_out.data = json.dumps(result)
            self.pub_obstacles.publish(msg_out)

            self.get_logger().debug(f"Detected {len(infos)} objects.")

        except Exception as e:
            self.get_logger().error(f"Error in lidar_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = UrbanLidarDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
