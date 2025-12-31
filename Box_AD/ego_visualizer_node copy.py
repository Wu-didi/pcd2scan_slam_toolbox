#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ego_visualizer_node.py

功能：
- 从 map.yaml 解析 PNG 左下角 UTM: (E_png, N_png)
- 订阅 /map (OccupancyGrid)，获取 map 原点 (origin_x, origin_y)
- 计算 map 坐标原点对应的 UTM:
      utm_origin_e = E_png - origin_x
      utm_origin_n = N_png - origin_y
- 订阅 /pose_utm (PoseStamped，frame_id="utm"，position.x=E，position.y=N)
- 在 map 坐标系下发布一个 CUBE Marker 表示自车位置和朝向

RViz 使用：
- Fixed Frame 选 "map"
- 添加一个 Marker 显示，topic 选 "/ego_marker"
"""

import os
import re
from typing import Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class EgoVisualizerNode(Node):
    def __init__(self):
        super().__init__('ego_visualizer_node')

        # ------------ 参数 ------------
        self.declare_parameter('map_yaml_path', '/home/wudi/slam/get_pc_from_db3/map.yaml')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('pose_utm_topic', 'pose_utm')

        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('utm_frame_id', 'utm')

        # 车辆尺寸和颜色（可按真实车身调整）
        self.declare_parameter('car_length', 2.0)
        self.declare_parameter('car_width', 1.0)
        self.declare_parameter('car_height', 0.75)

        self.declare_parameter('color_r', 0.0)
        self.declare_parameter('color_g', 1.0)
        self.declare_parameter('color_b', 1.0)
        self.declare_parameter('color_a', 0.9)

        self.map_yaml_path = self.get_parameter('map_yaml_path').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').value
        self.pose_utm_topic = self.get_parameter('pose_utm_topic').value

        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.utm_frame_id = self.get_parameter('utm_frame_id').value

        self.car_length = float(self.get_parameter('car_length').value)
        self.car_width = float(self.get_parameter('car_width').value)
        self.car_height = float(self.get_parameter('car_height').value)

        self.color_r = float(self.get_parameter('color_r').value)
        self.color_g = float(self.get_parameter('color_g').value)
        self.color_b = float(self.get_parameter('color_b').value)
        self.color_a = float(self.get_parameter('color_a').value)

        # ------------ 从 map.yaml 解析 PNG origin UTM ------------
        self.png_origin_e: Optional[float] = None
        self.png_origin_n: Optional[float] = None
        self.load_png_origin_utm(self.map_yaml_path)

        # ------------ /map 原点 & utm_origin ------------
        self.map_origin_x: Optional[float] = None
        self.map_origin_y: Optional[float] = None
        self.utm_origin_e: Optional[float] = None
        self.utm_origin_n: Optional[float] = None

        self._utm_warned = False  # 防止日志刷屏

        # ------------ 订阅 /map ------------
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )

        # ------------ 订阅 /pose_utm ------------
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_utm_topic,
            self.pose_utm_callback,
            10
        )

        # ------------ 发布 Marker ------------
        self.marker_pub = self.create_publisher(
            Marker,
            'ego_marker',
            10
        )

        self.get_logger().info(
            f"EgoVisualizerNode 启动：\n"
            f"  map_yaml_path = {self.map_yaml_path}\n"
            f"  map_topic     = {self.map_topic}\n"
            f"  pose_utm_topic = {self.pose_utm_topic}\n"
            f"  输出 CUBE Marker: /ego_marker, frame_id={self.map_frame_id}"
        )

    # ================== 解析 map.yaml ==================

    def load_png_origin_utm(self, yaml_path: str):
        """
        从 map.yaml 中解析:
            # PNG origin UTM: E=..., N=...
        得到 PNG 左下角对应的 UTM 坐标 (png_origin_e, png_origin_n)。
        """
        if not os.path.exists(yaml_path):
            self.get_logger().error(f"map_yaml_path 不存在: {yaml_path}")
            return

        try:
            with open(yaml_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# PNG origin UTM"):
                        m = re.search(r"E\s*=\s*([0-9.+\-eE]+).*N\s*=\s*([0-9.+\-eE]+)", line)
                        if m:
                            self.png_origin_e = float(m.group(1))
                            self.png_origin_n = float(m.group(2))
                            self.get_logger().info(
                                f"从 map.yaml 解析 PNG origin UTM: "
                                f"E={self.png_origin_e:.3f}, N={self.png_origin_n:.3f}"
                            )
                            return
            self.get_logger().warn(
                "在 map.yaml 中未找到 '# PNG origin UTM: E=..., N=...' 注释，"
                "将无法计算 utm_origin_e/n，自车位置可视化会不准确。"
            )
        except Exception as e:
            self.get_logger().error(f"解析 map.yaml 失败: {e}")

    # ================== /map 回调：计算 utm_origin ==================

    def map_callback(self, msg: OccupancyGrid):
        """
        OccupancyGrid.info.origin.position 给出 PNG 左下角在 map 坐标系中的位置 (origin_x, origin_y)。

        已知:
            PNG 左下角 UTM: (E_png, N_png) = (png_origin_e, png_origin_n)
            PNG 左下角 map: (origin_x, origin_y)
        又有:
            E_png = utm_origin_e + origin_x
            N_png = utm_origin_n + origin_y
        => utm_origin_e = E_png - origin_x
           utm_origin_n = N_png - origin_y
        """
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        self.get_logger().info(
            f"收到 /map: origin=({self.map_origin_x:.3f}, {self.map_origin_y:.3f}), "
            f"res={msg.info.resolution:.3f}, size=({msg.info.width}x{msg.info.height})"
        )

        if self.png_origin_e is not None and self.png_origin_n is not None:
            self.utm_origin_e = self.png_origin_e - self.map_origin_x
            self.utm_origin_n = self.png_origin_n - self.map_origin_y
            self.get_logger().info(
                f"计算得到 utm_origin: E0={self.utm_origin_e:.3f}, N0={self.utm_origin_n:.3f} "
                f"(map (0,0) 对应的 UTM 坐标)"
            )
        else:
            self.get_logger().warn(
                "尚未解析到 PNG origin UTM，无法计算 utm_origin_e/n。"
            )

    # ================== UTM -> map ==================

    def utm_to_map_xy(self, E: float, N: float):
        """
        UTM (E, N) -> map (x, y)
        用：
            E = utm_origin_e + x_map
            N = utm_origin_n + y_map
        """
        if self.utm_origin_e is None or self.utm_origin_n is None:
            if not self._utm_warned:
                self.get_logger().warn(
                    "utm_origin_e/n 尚未就绪，暂时无法准确将 UTM 坐标转换到 map；"
                    "请确认 /map 已发布且 map.yaml 中含有 PNG origin UTM 注释。"
                )
                self._utm_warned = True
            # 临时直接返回 UTM 值，避免节点崩溃
            return E, N

        x = E - self.utm_origin_e
        y = N - self.utm_origin_n
        return x, y

    # ================== 生成车辆 CUBE Marker ==================

    def make_ego_marker(self, x_map: float, y_map: float, orientation) -> Marker:
        """
        在 map 坐标系下生成一个代表自车的立方体 Marker。
        orientation 直接用 UTM Pose 的 orientation（UTM->map 只有平移）。
        """
        m = Marker()
        m.header.frame_id = self.map_frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "ego_vehicle"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.scale.x = self.car_length
        m.scale.y = self.car_width
        m.scale.z = self.car_height

        m.color.r = self.color_r
        m.color.g = self.color_g
        m.color.b = self.color_b
        m.color.a = self.color_a

        m.pose.position.x = float(x_map)
        m.pose.position.y = float(y_map)
        m.pose.position.z = self.car_height * 0.5  # 让车“坐”在地面上

        m.pose.orientation = orientation
        return m

    # ================== /pose_utm 回调 ==================

    def pose_utm_callback(self, msg: PoseStamped):
        """
        /pose_utm：UTM 下自车位姿
        - position.x = E
        - position.y = N
        - orientation 为 yaw 对应的四元数
        """
        E = msg.pose.position.x
        N = msg.pose.position.y

        x_map, y_map = self.utm_to_map_xy(E, N)
        marker = self.make_ego_marker(x_map, y_map, msg.pose.orientation)
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = EgoVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
