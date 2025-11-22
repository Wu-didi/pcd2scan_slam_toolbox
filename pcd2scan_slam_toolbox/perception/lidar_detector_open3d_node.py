#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lidar_detector_oneframe_open3d_node.py

订阅 /lidar_point_prev (PointCloud2)，只处理第一帧：
- 按 urban_lidar_autoware_like.py 中的参数和流程做检测
- 使用 Open3D 显示点云 + 目标 OBB
- 关闭 Open3D 窗口后，节点退出（调参用）
"""

import math
from typing import List, Tuple

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
try:
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    from sensor_msgs import point_cloud2 as pc2


# ---------------- 工具函数 / 检测逻辑（与你原脚本一致） ----------------

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
    if sx >= sy:
        return sx, sy
    else:
        return sy, sx


def ray_ground_filter(pc: o3d.geometry.PointCloud,
                      n_sectors: int = 180,
                      min_range: float = 1.5,
                      max_range: float = 80.0,
                      max_slope_deg: float = 18.0,
                      seed_dist: float = 3.0,
                      height_thresh: float = 0.2,
                      seed_percentile: float = 10.0):
    if pc.is_empty():
        n = 0
        return np.zeros((n,), dtype=bool), np.zeros((n,), dtype=bool)

    pts = np.asarray(pc.points)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    sector = ((theta + np.pi) / (2 * np.pi) * n_sectors).astype(int)
    sector = np.clip(sector, 0, n_sectors - 1)

    valid = (r >= min_range) & (r <= max_range)
    ground = np.zeros(len(pts), dtype=bool)
    non_ground = np.zeros(len(pts), dtype=bool)
    slope_tan = math.tan(math.radians(max_slope_deg))

    for s in range(n_sectors):
        idx = np.where(valid & (sector == s))[0]
        if idx.size == 0:
            continue
        si = idx[np.argsort(r[idx])]
        rs = r[si]
        zs = z[si]

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
                ground_z = 0.9 * ground_z + 0.1 * z[pj]
                prev_r = rs[j]
            else:
                non_ground[pj] = True
                prev_r = rs[j]

    undecided = ~(ground | non_ground)
    if np.any(undecided):
        g2 = z < (np.percentile(z[valid], 10.0) + height_thresh)
        ground[undecided & g2] = True
        non_ground[undecided & (~g2)] = True

    return ground, non_ground


def adaptive_dbscan(pc: o3d.geometry.PointCloud,
                    r_bins: List[Tuple[float, float]],
                    eps_list: List[float],
                    min_points: int = 12):
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
        labels = np.array(sub.cluster_dbscan(eps=eps,
                                             min_points=min_points,
                                             print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            continue
        for k in range(labels.max() + 1):
            sub_idx = sel[np.where(labels == k)[0]]
            if sub_idx.size >= min_points:
                clusters.append(sub_idx)
    return clusters


def obb_xy_and_info(cluster: o3d.geometry.PointCloud):
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
    if (2.0 <= L <= 6.0) and (1.2 <= W <= 2.8) and (1.2 <= H <= 3.2):
        return "vehicle"
    if (1.0 <= L <= 2.5) and (0.4 <= W <= 1.2) and (1.2 <= H <= 2.2):
        return "cyclist"
    if (0.2 <= L <= 1.2) and (0.2 <= W <= 1.0) and (1.2 <= H <= 2.2):
        return "pedestrian"
    return "unknown"


def color_for_class(name: str):
    mp = {
        "vehicle":   (1.0, 0.0, 0.0),
        "cyclist":   (1.0, 0.6, 0.0),
        "pedestrian":(0.0, 0.7, 0.0),
        "unknown":   (0.0, 0.5, 1.0)
    }
    return mp.get(name, (0.0, 0.5, 1.0))


class DetectorConfig:
    """参数完全按你原脚本默认值来。"""
    def __init__(self):
        # ROI
        self.xmin = 0.5
        self.xmax = 60.0
        self.ymin = -15.0
        self.ymax = 15.0
        self.zmin = -3.0
        self.zmax = 3.0

        # 预处理
        self.voxel = 0.08
        self.ro_nb = 8
        self.ro_radius = 0.45

        # Ray-Ground
        self.sectors = 120
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
        self.min_pts = 6

        # 尺寸/距离约束
        self.min_det_dist = 1.0
        self.max_det_dist = 50.0
        self.min_L = 0.3
        self.max_L = 7.0
        self.min_W = 0.2
        self.max_W = 3.5
        self.min_H = 0.2
        self.max_H = 3.5

        self.keep_unselected = True
        self.x_offset = 0.0


def process_one(pc_in: o3d.geometry.PointCloud, cfg: DetectorConfig):
    pc = crop_roi(pc_in,
                  (cfg.xmin, cfg.xmax),
                  (cfg.ymin, cfg.ymax),
                  (cfg.zmin, cfg.zmax))

    if cfg.voxel > 1e-6:
        pc = pc.voxel_down_sample(cfg.voxel)

    if cfg.ro_nb > 0:
        pc = radius_outlier(pc, nb_points=cfg.ro_nb, radius=cfg.ro_radius)

    if pc.is_empty():
        return o3d.geometry.PointCloud(), [], [], pc if cfg.keep_unselected else pc

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

    fg = pc.select_by_index(np.where(nmask)[0])
    if fg.is_empty():
        vis_cloud = pc if cfg.keep_unselected else fg
        return fg, [], [], vis_cloud

    bins = [(0, cfg.mid_dist), (cfg.mid_dist, cfg.far_dist), (cfg.far_dist, 1e6)]
    epses = [cfg.eps_near, cfg.eps_mid, cfg.eps_far]
    clusters_idx = adaptive_dbscan(fg, bins, epses, min_points=cfg.min_pts)

    boxes, infos, merged = [], [], []

    for idx in clusters_idx:
        sub = fg.select_by_index(idx)
        obb, inf = obb_xy_and_info(sub)
        L, W, H = inf["size"]

        center = np.asarray(obb.center)
        dist = float(np.linalg.norm(center[:2]))
        if not (cfg.min_det_dist <= dist <= cfg.max_det_dist):
            continue

        if not (cfg.min_L <= L <= cfg.max_L and
                cfg.min_W <= W <= cfg.max_W and
                cfg.min_H <= H <= cfg.max_H):
            continue

        label = heuristic_classify(L, W, H)
        obb.color = color_for_class(label)
        inf.update({"id": len(infos), "label": label})

        inf["center"][0] = float(inf["center"][0] - cfg.x_offset)

        boxes.append(obb)
        infos.append(inf)
        merged.append(sub)

    if merged:
        fg_all = merged[0]
        for m in merged[1:]:
            fg_all += m
    else:
        fg_all = o3d.geometry.PointCloud()

    vis_cloud = pc if cfg.keep_unselected else fg_all
    return fg_all, boxes, infos, vis_cloud


def pointcloud2_to_o3d(pc_msg: PointCloud2) -> o3d.geometry.PointCloud:
    points = []
    for p in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    pc = o3d.geometry.PointCloud()
    if points:
        arr = np.asarray(points, dtype=np.float64)
        pc.points = o3d.utility.Vector3dVector(arr)
    return pc


# ---------------- 单帧调参节点 ----------------

class LidarDetectorOneFrameNode(Node):
    def __init__(self):
        super().__init__("lidar_detector_oneframe_open3d")

        self.cfg = DetectorConfig()
        self.got_frame = False  # 只处理第一帧

        self.sub = self.create_subscription(
            PointCloud2,
            "/rslidar_points_prev",
            self.lidar_callback,
            10
        )

        self.get_logger().info(
            "Waiting for ONE PointCloud2 on /lidar_point_prev ... "
            "Will run detection and show Open3D window."
        )

    def lidar_callback(self, msg: PointCloud2):
        if self.got_frame:
            # 已经处理过一帧，后面的忽略
            return

        self.got_frame = True
        self.get_logger().info("Received first pointcloud, running detection...")

        pc_o3d = pointcloud2_to_o3d(msg)
        if pc_o3d.is_empty():
            self.get_logger().warn("First pointcloud is empty, nothing to show.")
            return

        _, boxes, infos, vis_cloud = process_one(pc_o3d, self.cfg)
        self.get_logger().info(f"Detected {len(infos)} objects.")

        # 用阻塞方式打开 Open3D 窗口，直到你手动关闭
        geoms = []
        if not vis_cloud.is_empty():
            geoms.append(vis_cloud)
        geoms.extend(boxes)

        if geoms:
            o3d.visualization.draw_geometries(
                geoms,
                window_name="Urban LiDAR Detection (One Frame)",
                width=1280,
                height=720
            )
        else:
            self.get_logger().warn("No geometry to visualize.")

        # 窗口关掉后，节点就可以退出了
        self.get_logger().info("Open3D window closed, shutting down node.")
        # 直接在 main 里检测 got_frame 并退出，这里不再做额外动作


def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectorOneFrameNode()

    # 自己管理循环：收到一帧并处理后退出
    try:
        while rclpy.ok() and not node.got_frame:
            rclpy.spin_once(node, timeout_sec=0.1)
        # 如果第一帧已经处理完（包括 Open3D 窗口关闭），就结束
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
