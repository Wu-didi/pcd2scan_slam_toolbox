#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local path planner node.

订阅:
- /global_path (nav_msgs/Path): 全局路径
- 定位话题 (PoseStamped / PoseWithCovarianceStamped / Odometry)

输出:
- /local_path (nav_msgs/Path): 以机器人当前位置为起点的局部路径

局部路径通过裁剪全局路径，限制在一定前瞻距离内，并保留有限数量的路径点。
"""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path


class LocalPathPlannerNode(Node):
    def __init__(self):
        super().__init__('local_path_planner_node')

        # --------- 参数 ---------
        self.declare_parameter('global_path_topic', 'global_path')
        self.declare_parameter('local_path_topic', 'local_path')
        self.declare_parameter('pose_topic', 'amcl_pose')
        self.declare_parameter('pose_type', 'PoseWithCovarianceStamped')
        self.declare_parameter('lookahead_distance', 8.0)
        self.declare_parameter('max_local_points', 50)
        self.declare_parameter('goal_tolerance', 0.5)

        self.global_path_topic = self.get_parameter('global_path_topic').value
        self.local_path_topic = self.get_parameter('local_path_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.pose_type = str(self.get_parameter('pose_type').value).lower()
        self.lookahead_distance = float(
            self.get_parameter('lookahead_distance').value)
        self.max_local_points = max(
            2, int(self.get_parameter('max_local_points').value))
        self.goal_tolerance = float(
            self.get_parameter('goal_tolerance').value)

        # --------- 状态 ---------
        self.global_path_msg: Optional[Path] = None
        self.global_xy: List[Tuple[float, float]] = []
        self.global_frame_id: str = 'map'

        self.current_pose: Optional[Pose] = None
        self.pose_frame_id: str = ''

        self.closest_idx_hint: int = 0
        self.warned_frame_mismatch = False

        # --------- 订阅 /global_path ---------
        global_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.global_path_sub = self.create_subscription(
            Path,
            self.global_path_topic,
            self.global_path_callback,
            global_qos,
        )

        # --------- 订阅定位话题 ---------
        self.pose_sub = self.create_pose_subscription()

        # --------- 发布 /local_path ---------
        self.local_path_pub = self.create_publisher(
            Path,
            self.local_path_topic,
            10
        )

        self.get_logger().info(
            f"LocalPathPlannerNode 启动，等待 {self.global_path_topic} 和 {self.pose_topic}")

    # ================== 订阅回调 ==================

    def global_path_callback(self, msg: Path):
        self.global_path_msg = msg
        self.global_frame_id = msg.header.frame_id or 'map'
        self.global_xy = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
        self.closest_idx_hint = 0

        self.get_logger().info(
            f"收到新的 global_path: {len(self.global_xy)} points, frame={self.global_frame_id}")

        if len(self.global_xy) < 2:
            self.get_logger().warn("global_path 点数 < 2，无法构建局部路径")
            return

        self.try_publish_local_path()

    def pose_stamped_callback(self, msg: PoseStamped):
        self.current_pose = self.copy_pose(msg.pose)
        self.pose_frame_id = msg.header.frame_id
        self.try_publish_local_path()

    def pose_with_cov_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = self.copy_pose(msg.pose.pose)
        self.pose_frame_id = msg.header.frame_id
        self.try_publish_local_path()

    def odom_callback(self, msg: Odometry):
        self.current_pose = self.copy_pose(msg.pose.pose)
        self.pose_frame_id = msg.header.frame_id
        self.try_publish_local_path()

    # ================== 辅助方法 ==================

    def create_pose_subscription(self):
        if 'cov' in self.pose_type:
            return self.create_subscription(
                PoseWithCovarianceStamped,
                self.pose_topic,
                self.pose_with_cov_callback,
                10
            )
        if 'odom' in self.pose_type:
            return self.create_subscription(
                Odometry,
                self.pose_topic,
                self.odom_callback,
                10
            )
        return self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_stamped_callback,
            10
        )

    @staticmethod
    def copy_pose(src: Pose) -> Pose:
        dst = Pose()
        dst.position.x = src.position.x
        dst.position.y = src.position.y
        dst.position.z = src.position.z
        dst.orientation.x = src.orientation.x
        dst.orientation.y = src.orientation.y
        dst.orientation.z = src.orientation.z
        dst.orientation.w = src.orientation.w
        return dst

    def try_publish_local_path(self):
        if self.global_path_msg is None or not self.global_xy:
            return
        if self.current_pose is None:
            return

        pose_frame = self.pose_frame_id or 'map'
        if pose_frame != self.global_frame_id and not self.warned_frame_mismatch:
            self.warned_frame_mismatch = True
            self.get_logger().warn(
                f"定位话题 frame={pose_frame} 与 global_path frame={self.global_frame_id} 不一致，假定已经在同一坐标系")

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        idx = self.find_closest_index(robot_x, robot_y)
        if idx is None:
            return

        # 如果距离终点很近，直接输出到终点
        goal_x, goal_y = self.global_xy[-1]
        goal_dist = math.hypot(robot_x - goal_x, robot_y - goal_y)
        if goal_dist <= self.goal_tolerance:
            idx = len(self.global_xy) - 1

        local_poses = self.build_local_segment(idx)
        if len(local_poses) < 2:
            return

        local_path = Path()
        local_path.header.frame_id = self.global_frame_id
        local_path.header.stamp = self.get_clock().now().to_msg()
        local_path.poses = local_poses

        self.local_path_pub.publish(local_path)

    def find_closest_index(self, x: float, y: float) -> Optional[int]:
        if not self.global_xy:
            return None

        start_idx = max(self.closest_idx_hint - 5, 0)
        best_idx = start_idx
        best_dist = float('inf')

        for idx in range(start_idx, len(self.global_xy)):
            px, py = self.global_xy[idx]
            dist = (px - x) * (px - x) + (py - y) * (py - y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
            # 如果离当前点非常近，就提前终止搜索
            if dist <= 0.04:  # 20cm
                break

        self.closest_idx_hint = best_idx
        return best_idx

    def build_local_segment(self, start_idx: int) -> List[PoseStamped]:
        now = self.get_clock().now().to_msg()
        poses: List[PoseStamped] = []

        start_pose = PoseStamped()
        start_pose.header.frame_id = self.global_frame_id
        start_pose.header.stamp = now
        start_pose.pose = self.copy_pose(self.current_pose)
        poses.append(start_pose)

        last_x = self.current_pose.position.x
        last_y = self.current_pose.position.y
        travelled = 0.0

        for idx in range(start_idx, len(self.global_path_msg.poses)):
            gp = self.global_path_msg.poses[idx]
            pose_copy = PoseStamped()
            pose_copy.header.frame_id = self.global_frame_id
            pose_copy.header.stamp = now
            pose_copy.pose = self.copy_pose(gp.pose)
            poses.append(pose_copy)

            px, py = pose_copy.pose.position.x, pose_copy.pose.position.y
            travelled += math.hypot(px - last_x, py - last_y)
            last_x, last_y = px, py

            if travelled >= self.lookahead_distance:
                break
            if len(poses) >= self.max_local_points:
                break

        return poses


def main(args=None):
    rclpy.init(args=args)
    node = LocalPathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
