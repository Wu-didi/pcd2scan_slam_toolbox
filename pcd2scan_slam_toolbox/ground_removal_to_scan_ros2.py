#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ground_removal_to_scan_ros2.py

功能：
- 读取单个点云文件或目录中的多个点云文件（pcd/ply/bin/npy/...）
- 执行 Ray-Ground 去地面
- 将非地面点投影到2D平面，按角度排序，封装成 LaserScan 并发布到 /scan
- 额外：把原始点云发布成 PointCloud2 到 /raw_cloud，方便 RViz 对照
- 额外：把去地面后的前景点发布到 /nonground_cloud（可选看）
"""

import os
import sys
import glob
import math
import argparse
from typing import List, Tuple

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
import struct


# ===================== 基础 I/O =====================

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pcd", ".ply", ".xyz", ".xyzn", ".xyzrgb", ".pts"]:
        pc = o3d.io.read_point_cloud(path)
        if pc.is_empty():
            raise RuntimeError(f"空点云: {path}")
        return pc
    elif ext == ".bin":
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 4 == 0:
            arr = arr.reshape(-1, 4)
        else:
            arr = arr.reshape(-1, 3)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(arr[:, :3])
        return pc
    elif ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise RuntimeError(f"npy 格式需 (N,>=3): {path}")
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(arr[:, :3])
        return pc
    else:
        raise RuntimeError(f"不支持格式: {ext}")


def list_point_cloud_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        exts = [".pcd", ".ply", ".xyz", ".xyzn", ".xyzrgb", ".pts", ".bin", ".npy"]
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(input_path, f"*{e}")))
        files.sort()
        return files
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        raise FileNotFoundError(f"输入不存在: {input_path}")


# ===================== Ray-Ground Filter =====================

def ray_ground_filter(
    pts_xyz: np.ndarray,
    n_sectors: int = 180,
    min_range: float = 1.5,
    max_range: float = 80.0,
    max_slope_deg: float = 18.0,
    seed_dist: float = 3.0,
    height_thresh: float = 0.2,
    seed_percentile: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回: ground_mask, non_ground_mask
    """
    if pts_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)

    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]
    z = pts_xyz[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    sector = ((theta + np.pi) / (2 * np.pi) * n_sectors).astype(int)
    sector = np.clip(sector, 0, n_sectors - 1)

    valid = (r >= min_range) & (r <= max_range)
    ground = np.zeros(len(pts_xyz), dtype=bool)
    non_ground = np.zeros(len(pts_xyz), dtype=bool)
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

    # 兜底：还没分出来的点按高度再分一次
    undecided = ~(ground | non_ground)
    if np.any(undecided):
        g2 = z < (np.percentile(z[valid], 10.0) + height_thresh)
        ground[undecided & g2] = True
        non_ground[undecided & (~g2)] = True

    return ground, non_ground


# ===================== 2D -> LaserScan =====================

def xy_to_laserscan_msg(
    xy: np.ndarray,
    frame_id: str = "laser",
    angle_min: float = -math.pi,
    angle_max: float = math.pi,
    angle_increment: float = math.radians(0.5),
    range_min: float = 0.05,
    range_max: float = 100.0,
) -> LaserScan:
    if angle_increment <= 0.0:
        angle_increment = math.radians(0.5)

    num_beams = int((angle_max - angle_min) / angle_increment) + 1
    if num_beams > 2000:
        num_beams = 2000

    ranges = np.full(num_beams, np.inf, dtype=np.float32)

    if xy.shape[0] > 0:
        xs = xy[:, 0]
        ys = xy[:, 1]
        angles = np.arctan2(ys, xs)
        dists = np.sqrt(xs * xs + ys * ys)

        for a, r in zip(angles, dists):
            if r < range_min or r > range_max:
                continue
            idx = int((a - angle_min) / angle_increment)
            if 0 <= idx < num_beams:
                if r < ranges[idx]:
                    ranges[idx] = r

    ranges = np.where(np.isinf(ranges), range_max, ranges)

    msg = LaserScan()
    msg.header.frame_id = frame_id
    msg.angle_min = angle_min
    msg.angle_max = angle_max
    msg.angle_increment = angle_increment
    msg.range_min = range_min
    msg.range_max = range_max
    msg.ranges = ranges.astype(float).tolist()
    return msg


# ===================== 点云 -> PointCloud2 =====================

def ndarray_to_pointcloud2(points: np.ndarray, frame_id: str) -> PointCloud2:
    """
    points: (N,3) float32 or float64
    """
    # 确保是 float32
    pts = points.astype(np.float32)
    header = Header()
    header.stamp = rclpy.time.Time().to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12  # 3 * 4 bytes
    data = pts.tobytes()

    msg = PointCloud2(
        header=header,
        height=1,
        width=pts.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=point_step,
        row_step=point_step * pts.shape[0],
        data=data,
    )
    return msg


# ===================== ROS2 Node =====================

class GroundRemovalScanPublisher(Node):
    def __init__(self, args):
        super().__init__("ground_removal_scan_publisher")
        self.args = args
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        # 新增：原始点云
        self.raw_pub = self.create_publisher(PointCloud2, "/raw_cloud", 5)
        # 新增：去地面后的点（可选）
        self.nonground_pub = self.create_publisher(PointCloud2, "/nonground_cloud", 5)

        # 预加载所有文件路径
        self.files = list_point_cloud_files(args.input)
        if not self.files:
            raise RuntimeError("未找到点云文件")
        self.get_logger().info(f"发现 {len(self.files)} 个点云文件.")

        self.file_idx = 0  # 当前播到哪个文件

        # 发布频率
        period = 1.0 / float(args.pub_rate)
        self.timer = self.create_timer(period, self.timer_cb)

    def timer_cb(self):
        path = self.files[self.file_idx]
        try:
            pc = load_point_cloud(path)
            pts = np.asarray(pc.points)  # (N,3)
            # 先把原始点云发出去
            if pts.shape[0] > 0:
                raw_msg = ndarray_to_pointcloud2(pts, self.args.frame_id)
                raw_msg.header.stamp = self.get_clock().now().to_msg()
                self.raw_pub.publish(raw_msg)

            if pts.shape[0] == 0:
                self.get_logger().warn(f"{os.path.basename(path)} 是空点云")
                xy_sorted = np.zeros((0, 2), dtype=np.float32)
                nonground_pts = np.zeros((0, 3), dtype=np.float32)
            else:
                gmask, nmask = ray_ground_filter(
                    pts,
                    n_sectors=self.args.sectors,
                    min_range=self.args.min_range,
                    max_range=self.args.max_range,
                    max_slope_deg=self.args.max_slope_deg,
                    seed_dist=self.args.seed_dist,
                    height_thresh=self.args.height_thresh,
                    seed_percentile=self.args.seed_pct,
                )
                nonground_pts = pts[nmask]
                if nonground_pts.shape[0] == 0:
                    xy_sorted = np.zeros((0, 2), dtype=np.float32)
                else:
                    # 发去地面的点
                    ng_msg = ndarray_to_pointcloud2(nonground_pts, self.args.frame_id)
                    ng_msg.header.stamp = self.get_clock().now().to_msg()
                    self.nonground_pub.publish(ng_msg)

                    xy = nonground_pts[:, :2]
                    ang = np.arctan2(xy[:, 1], xy[:, 0])
                    order = np.argsort(ang)
                    xy_sorted = xy[order]

            # 发布 scan
            scan_msg = xy_to_laserscan_msg(
                xy_sorted,
                frame_id=self.args.frame_id,
                angle_min=self.args.angle_min,
                angle_max=self.args.angle_max,
                angle_increment=self.args.angle_inc,
                range_min=self.args.range_min,
                range_max=self.args.range_max,
            )
            scan_msg.header.stamp = self.get_clock().now().to_msg()
            self.scan_pub.publish(scan_msg)
            self.get_logger().info(
                f"发布 /scan 来自 {os.path.basename(path)} ，beams={len(scan_msg.ranges)}"
            )

        except Exception as e:
            self.get_logger().error(f"处理 {path} 出错: {e}")

        # 播下一张，循环
        self.file_idx = (self.file_idx + 1) % len(self.files)


# ===================== 命令行参数 =====================

def build_args():
    p = argparse.ArgumentParser("Ground removal -> LaserScan ROS2 publisher (loop)")
    p.add_argument(
        "--input",
        default="/home/wudi/slam/pcd2scan_slam_toolbox/pcd2scan_slam_toolbox/data/1724726935637000000.pcd",
        help="输入点云文件或目录",
    )

    # Ray-Ground 参数
    p.add_argument("--sectors", type=int, default=180)
    p.add_argument("--min_range", type=float, default=1.5)
    p.add_argument("--max_range", type=float, default=80.0)
    p.add_argument("--max_slope_deg", type=float, default=18.0)
    p.add_argument("--seed_dist", type=float, default=3.0)
    p.add_argument("--height_thresh", type=float, default=0.2)
    p.add_argument("--seed_pct", type=float, default=10.0)

    # LaserScan 参数
    p.add_argument("--frame-id", default="laser")
    p.add_argument("--angle-min", type=float, default=-math.pi)
    p.add_argument("--angle-max", type=float, default=math.pi)
    p.add_argument("--angle-inc", type=float, default=math.radians(0.5))
    p.add_argument("--range-min", type=float, default=0.05)
    p.add_argument("--range-max", type=float, default=100.0)

    # 发布频率
    p.add_argument("--pub-rate", type=float, default=10.0, help="发布频率Hz，默认10")

    return p.parse_args()


# ===================== main =====================

def main():
    args = build_args()
    rclpy.init()
    node = GroundRemovalScanPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
