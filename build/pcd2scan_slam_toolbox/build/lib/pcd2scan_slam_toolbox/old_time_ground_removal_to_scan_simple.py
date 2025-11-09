#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ground_removal_to_scan_simple.py

简化版：
- 订阅实时点云（比如 /rslidar_points_prev）
- 仅按高度阈值去地面（不做扇区坡度判断）
- 投影到2D平面，按角度排序，封装成 LaserScan 发布到 /scan
- 额外发布 /raw_cloud 和 /nonground_cloud 方便 RViz 查看
"""

import math
import argparse
from typing import Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header


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


def ndarray_to_pointcloud2(points: np.ndarray, frame_id: str, stamp=None) -> PointCloud2:
    pts = points.astype(np.float32)
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp  # ← stamp从外面传进来

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


class GroundRemovalScanPublisher(Node):
    def __init__(self, args):
        super().__init__("ground_removal_scan_publisher")
        self.args = args

        # 发布者
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.raw_pub = self.create_publisher(PointCloud2, "/raw_cloud", 5)
        self.nonground_pub = self.create_publisher(PointCloud2, "/nonground_cloud", 5)

        # 订阅点云
        self.sub = self.create_subscription(
            PointCloud2,
            args.input_topic,
            self.cloud_cb,
            10
        )

        self.get_logger().info(f"订阅点云: {args.input_topic}")
        # 高度阈值，可根据安装位置改
        self.z_thresh = args.z_thresh

    def cloud_cb(self, msg: PointCloud2):
        # ★ 用点云自带的时间戳，而不是当前时间
        cloud_stamp = msg.header.stamp

        pts = self.pointcloud2_to_xyz(msg)

        # 原始点云转发
        if pts.shape[0] > 0:
            raw_msg = ndarray_to_pointcloud2(pts, self.args.frame_id, stamp=cloud_stamp)
            self.raw_pub.publish(raw_msg)

        if pts.shape[0] == 0:
            xy_sorted = np.zeros((0, 2), dtype=np.float32)
            nonground_pts = np.zeros((0, 3), dtype=np.float32)
        else:
            # 高度去地面
            mask = pts[:, 2] > self.z_thresh
            nonground_pts = pts[mask]

            if nonground_pts.shape[0] > 0:
                ng_msg = ndarray_to_pointcloud2(nonground_pts, self.args.frame_id, stamp=cloud_stamp)
                self.nonground_pub.publish(ng_msg)

                # 投影并排序
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
        # ★ /scan 也用点云的时间戳
        scan_msg.header.stamp = cloud_stamp
        self.scan_pub.publish(scan_msg)

        self.get_logger().info(
            f"发布 /scan beams={len(scan_msg.ranges)} pts={pts.shape[0]} fg={nonground_pts.shape[0]}"
        )

    def pointcloud2_to_xyz(self, cloud: PointCloud2) -> np.ndarray:
        """简单版 PointCloud2 -> (N,3)"""
        if cloud.width == 0:
            return np.zeros((0, 3), dtype=np.float32)

        import struct
        pts = []
        fmt = "<fff"
        step = cloud.point_step
        data = cloud.data

        for i in range(cloud.width * cloud.height):
            offset = i * step
            x, y, z = struct.unpack_from(fmt, data, offset)
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                pts.append((x, y, z))

        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(pts, dtype=np.float32)


def build_args():
    p = argparse.ArgumentParser("Height removal -> LaserScan ROS2 subscriber")
    p.add_argument("--input-topic",
                   default="/rslidar_points_prev",
                   help="要订阅的点云topic，比如 /rslidar_points_prev")

    p.add_argument("--z-thresh", type=float, default=0.5)

    # LaserScan 参数
    p.add_argument("--frame-id", default="laser")
    p.add_argument("--angle-min", type=float, default=-math.pi)
    p.add_argument("--angle-max", type=float, default=math.pi)
    p.add_argument("--angle-inc", type=float, default=math.radians(1))
    p.add_argument("--range-min", type=float, default=0.05)
    p.add_argument("--range-max", type=float, default=100.0)

    args = p.parse_args()
    return args


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
