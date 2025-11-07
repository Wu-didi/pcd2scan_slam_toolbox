#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ground_removal_to_scan_ros2.py

功能：
- 订阅实时点云（来自一个激光雷达 topic，比如 /rslidar_points_prev）
- 执行 Ray-Ground 去地面
- 将非地面点投影到2D平面，按角度排序，封装成 LaserScan 并发布到 /scan
- 额外：把原始点云发布成 PointCloud2 到 /raw_cloud，方便 RViz 对照
- 额外：把去地面后的前景点发布到 /nonground_cloud
"""

import math
import argparse
from typing import Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header

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

    # 兜底：没分出来的按高度粗分
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
    # 防止太大
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


# ===================== ndarray -> PointCloud2 =====================

def ndarray_to_pointcloud2(points: np.ndarray, frame_id: str, stamp=None) -> PointCloud2:
    pts = points.astype(np.float32)
    header = Header()
    header.frame_id = frame_id
    
    # 后续在车上实际使用时，取消下面的注释，使用传进来的时间戳
    # if stamp is not None:
    #     header.stamp = stamp
        
    header.stamp = stamp   # 强制用传进来的时间

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12
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

        # 发布者
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.raw_pub = self.create_publisher(PointCloud2, "/raw_cloud", 5)
        self.nonground_pub = self.create_publisher(PointCloud2, "/nonground_cloud", 5)

        # 订阅一个在线点云topic
        self.sub = self.create_subscription(
            PointCloud2,
            args.input_topic,
            self.cloud_cb,
            10
        )

        self.get_logger().info(f"订阅点云: {args.input_topic}")

    def cloud_cb(self, msg: PointCloud2):
        # msg 是 sensor_msgs/PointCloud2
        # 把 PointCloud2 转成 numpy (N,3)
        
        # ⭐ 关键：我们不用 msg.header.stamp，当成“现在”
        now_stamp = self.get_clock().now().to_msg()

        
        pts = self.pointcloud2_to_xyz(msg)
        # stamp = msg.header.stamp

        # 原始点云直接转发一下，方便 rviz 看
        if pts.shape[0] > 0:
            raw_msg = ndarray_to_pointcloud2(pts, self.args.frame_id, stamp=now_stamp)
            self.raw_pub.publish(raw_msg)

        if pts.shape[0] == 0:
            xy_sorted = np.zeros((0, 2), dtype=np.float32)
            nonground_pts = np.zeros((0, 3), dtype=np.float32)
        else:
            # 去地面
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

            if nonground_pts.shape[0] > 0:
                ng_msg = ndarray_to_pointcloud2(nonground_pts, self.args.frame_id, stamp=now_stamp)
                self.nonground_pub.publish(ng_msg)

                # 投影到 xy 并排序
                xy = nonground_pts[:, :2]
                ang = np.arctan2(xy[:, 1], xy[:, 0])
                order = np.argsort(ang)
                xy_sorted = xy[order]
            else:
                xy_sorted = np.zeros((0, 2), dtype=np.float32)

        # 发布 /scan
        scan_msg = xy_to_laserscan_msg(
            xy_sorted,
            frame_id=self.args.frame_id,
            angle_min=self.args.angle_min,
            angle_max=self.args.angle_max,
            angle_increment=self.args.angle_inc,
            range_min=self.args.range_min,
            range_max=self.args.range_max,
        )
        # scan_msg.header.stamp = stamp
        scan_msg.header.stamp = now_stamp   # ⭐ 用现在的时间

        self.scan_pub.publish(scan_msg)
        self.get_logger().info(f"发布 /scan beams={len(scan_msg.ranges)}   pts={pts.shape[0]}")

    # ===== PointCloud2 -> numpy =====
    def pointcloud2_to_xyz(self, cloud: PointCloud2) -> np.ndarray:
        """
        简单版 PointCloud2 -> (N,3)，假设字段里有 x,y,z 且是 float32
        """
        if cloud.width == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # sensor_msgs_py 也可以用；这里手写解析更独立
        import struct

        pts = []
        fmt = "<fff"   # x,y,z float32
        step = cloud.point_step
        data = cloud.data

        for i in range(cloud.width * cloud.height):
            offset = i * step
            x, y, z = struct.unpack_from(fmt, data, offset)
            # 可以在这里过滤无效点
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                pts.append((x, y, z))

        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(pts, dtype=np.float32)


# ===================== 命令行参数 =====================

def build_args():
    p = argparse.ArgumentParser("Ground removal -> LaserScan ROS2 subscriber")
    # 这里变成订阅 topic，不再是读文件
    p.add_argument("--input-topic",
                   default="/rslidar_points_prev",
                   help="要订阅的点云topic，比如 /rslidar_points_prev")

    # Ray-Ground 参数
    p.add_argument("--sectors", type=int, default=180)
    p.add_argument("--min-range", type=float, default=1.5)
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--max-slope-deg", type=float, default=18.0)
    p.add_argument("--seed-dist", type=float, default=3.0)
    p.add_argument("--height-thresh", type=float, default=0.2)
    p.add_argument("--seed-pct", type=float, default=10.0)

    # LaserScan 参数
    p.add_argument("--frame-id", default="laser")
    p.add_argument("--angle-min", type=float, default=-math.pi)
    p.add_argument("--angle-max", type=float, default=math.pi)
    p.add_argument("--angle-inc", type=float, default=math.radians(0.5))
    p.add_argument("--range-min", type=float, default=0.05)
    p.add_argument("--range-max", type=float, default=100.0)

    args = p.parse_args()

    # argparse 会把带 - 的字段变成下划线属性，这里做一下对齐
    args.min_range = args.min_range
    args.max_range = args.max_range
    args.max_slope_deg = args.max_slope_deg
    args.seed_dist = args.seed_dist
    args.height_thresh = args.height_thresh
    args.seed_pct = args.seed_pct
    args.angle_min = args.angle_min
    args.angle_max = args.angle_max
    args.angle_inc = args.angle_inc
    args.range_min = args.range_min
    args.range_max = args.range_max

    return args


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
