#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fast_obstacle_viz_node.py

可视化 fast_lidar_obstacle_detector 的输出：
- 订阅 /fast_obstacles (std_msgs/String, JSON)
- 将每个 obstacle 映射为一个 Marker（小立方体或球）
- 发布 /fast_obstacles_markers (visualization_msgs/MarkerArray)
"""

import json
import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion


def yaw_to_quat(yaw: float) -> Quaternion:
    """只有 yaw 的简单四元数，虽然这里不太用得到。"""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class FastObstacleVizNode(Node):
    def __init__(self):
        super().__init__("fast_obstacle_viz")

        # 可视化相关参数（你可以根据 grid_resolution 调整）
        self.marker_scale_xy = 0.5   # x,y 方向大小（米）
        self.marker_scale_z = 0.5    # z 方向大小（米）
        self.marker_alpha = 0.6      # 透明度

        # 订阅 fast_obstacles
        self.sub = self.create_subscription(
            String,
            "/fast_obstacles",
            # "fast_obstacles_stable",
            self.obstacles_callback,
            10
        )

        # 发布 MarkerArray
        self.pub = self.create_publisher(
            MarkerArray,
            "/fast_obstacles_markers",
            10
        )

        self.get_logger().info(
            "FastObstacleVizNode started. "
            "Subscribe /fast_obstacles, publish /fast_obstacles_markers"
        )

    def obstacles_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse JSON from /fast_obstacles: {e}")
            return

        obstacles = data.get("obstacles", [])
        frame_id = data.get("frame_id", "lidar")

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # 给每一帧一个新的 namespace+id 集合，通过 lifetime 自动清掉旧的
        for i, obj in enumerate(obstacles):
            try:
                x = float(obj["x"])
                y = float(obj["y"])
                z = float(obj["z"])
                count = int(obj.get("count", 0))
            except KeyError as e:
                self.get_logger().warn(f"Missing key in obstacle: {e}")
                continue

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = "fast_obstacles"
            m.id = i
            m.type = Marker.CUBE          # 或 Marker.SPHERE，看你喜好
            m.action = Marker.ADD

            # 位置
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z

            # 没有方向信息，就用单位四元数
            m.pose.orientation.w = 1.0
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0

            # 大小：可和 fast 节点里的 grid_resolution 对齐
            m.scale.x = self.marker_scale_xy
            m.scale.y = self.marker_scale_xy
            m.scale.z = self.marker_scale_z

            # 颜色：点多的格子颜色亮一点/偏红一点（可选简单映射）
            # 这里用一个简单归一化：count 越多越接近红色
            # t = max(0.0, min(1.0, count / 20.0))  # 20 个点及以上算“满”
            # m.color.r = float(0.2 + 0.8 * t)      # 0.2~1.0
            # m.color.g = float(0.8 * (1.0 - t))    # 0.8~0
            # m.color.b = 0.2
            # m.color.a = self.marker_alpha
            

            # t 还是代表密度，0~1
            t = max(0.0, min(1.0, count / 20.0))  # 20 个点及以上算“满”

            # 低密度：偏青绿，高密度：偏深蓝
            m.color.r = 0.0
            m.color.g = float(0.8 * (1.0 - t))   # 0.8 -> 0
            m.color.b = float(0.5 + 0.5 * t)     # 0.5 -> 1.0
            m.color.a = self.marker_alpha


            # 生命周期，避免残留
            m.lifetime.sec = 0
            m.lifetime.nanosec = int(0.3 * 1e9)  # 0.3 秒

            marker_array.markers.append(m)

        # 如果没有障碍物，也发一个空 MarkerArray，让 RViz 按 lifetime 自然清除旧 marker
        self.pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = FastObstacleVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
