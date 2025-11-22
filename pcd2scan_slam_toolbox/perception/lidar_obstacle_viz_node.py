#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lidar_obstacle_viz_node.py

功能：
- 订阅 /lidar_obstacles (std_msgs/String)，内容为 JSON：
  {
    "stamp": {"sec": ..., "nanosec": ...},
    "frame_id": "lidar",
    "objects": [
      {
        "id": 0,
        "label": "vehicle",
        "center": [x, y, z],
        "size": [L, W, H],
        "yaw": yaw
      }, ...
    ]
  }

- 将每个 object 转为一个 Marker（CUBE），发布到 /lidar_obstacles_markers
- 在 RViz 中添加 MarkerArray 显示，就可以看到 3D 障碍物框
"""

import math
import json

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import String
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """仅绕 Z 轴的 yaw 角转成四元数"""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def color_for_label(label: str):
    """和检测节点保持一致的颜色风格"""
    if label == "vehicle":
        return 1.0, 0.0, 0.0  # 红
    if label == "cyclist":
        return 1.0, 0.6, 0.0  # 橙
    if label == "pedestrian":
        return 0.0, 0.7, 0.0  # 绿
    return 0.0, 0.5, 1.0      # 蓝（unknown）


class LidarObstacleVizNode(Node):
    def __init__(self):
        super().__init__("lidar_obstacle_viz")

        # 订阅检测结果
        self.sub = self.create_subscription(
            String,
            "/lidar_obstacles",
            self.obstacles_callback,
            10
        )

        # 发布 MarkerArray
        self.pub_markers = self.create_publisher(
            MarkerArray,
            "/lidar_obstacles_markers",
            10
        )

        # 可调：Marker 存活时间（避免旧框残留）
        self.marker_lifetime = Duration(seconds=0.3)

        self.get_logger().info(
            "LidarObstacleVizNode started. Subscribe /lidar_obstacles, "
            "publish /lidar_obstacles_markers"
        )

    def obstacles_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse JSON from /lidar_obstacles: {e}")
            return

        objects = data.get("objects", [])
        frame_id = data.get("frame_id", "lidar")

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # 为了可视化干净，id 从 0 连续编号即可
        for i, obj in enumerate(objects):
            try:
                center = obj["center"]      # [x, y, z]
                size = obj["size"]          # [L, W, H]
                yaw = float(obj["yaw"])     # rad
                label = obj.get("label", "unknown")
            except KeyError as e:
                self.get_logger().warn(f"Missing key in object: {e}")
                continue

            # 创建 CUBE marker
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = "lidar_obstacles"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD

            # 位姿：中心 + yaw
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation = yaw_to_quaternion(yaw)

            # 尺寸：L, W, H
            m.scale.x = float(size[0])
            m.scale.y = float(size[1])
            m.scale.z = float(size[2])

            # 颜色：按类别
            r, g, b = color_for_label(label)
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)
            m.color.a = 0.6  # 透明度

            # 生命周期：避免旧框残留
            m.lifetime = self.marker_lifetime.to_msg()

            marker_array.markers.append(m)

        # 如果你想在没有目标时也清空旧 Marker，可以在 objects 为空时发布一个空的 MarkerArray
        # RViz 会依靠 lifetime 将旧的全部清掉
        self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
