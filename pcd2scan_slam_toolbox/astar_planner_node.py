#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import heapq

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


class AstarPlannerNode(Node):
    """
    功能：
    - 订阅静态 /map（OccupancyGrid）
    - 订阅 /initialpose（起点） 和 /goal_pose（终点）
    - 只要起点和终点都已设置，每次收到新的起点或终点，都会：
        * 使用膨胀后的栅格地图做 A* 规划
        * 对路径做 line-of-sight 简化
        * 发布一条 /global_path
    """

    def __init__(self):
        super().__init__('astar_planner_node')

        # 机器人“半径”配置（单位米）：车宽的一半 + 安全距离
        self.declare_parameter('robot_radius', 0.5)
        self.robot_radius = float(self.get_parameter('robot_radius').value)

        # 地图数据
        self.map_received = False
        self.grid = None          # numpy int8: -1,0,100
        self.width = 0
        self.height = 0
        self.resolution = 0.1
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.frame_id = 'map'

        # 当前起点 / 终点（可以多次更新）
        self.start_pose = None
        self.goal_pose = None

        # ---- 订阅 /map ----
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )

        # ---- 订阅起点&终点 ----
        self.initpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'initialpose',
            self.initialpose_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_callback,
            10
        )

        # ---- 发布路径（使用 TRANSIENT_LOCAL，方便之后打开 RViz 也能看到最后一条）----
        path_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.path_pub = self.create_publisher(Path, 'global_path', path_qos)

        self.get_logger().info(
            "AstarPlannerNode 启动，等待 /map、/initialpose、/goal_pose ...")

    # ========== 话题回调 ==========

    def map_callback(self, msg: OccupancyGrid):
        # 原始 OccupancyGrid
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.frame_id = msg.header.frame_id

        data = np.array(msg.data, dtype=np.int8)
        grid = data.reshape((self.height, self.width))

        # ---- 障碍膨胀 ----
        obst = (grid == 100).astype(np.uint8)

        inflation_cells = int(self.robot_radius / self.resolution)
        inflation_cells = max(inflation_cells, 1)
        kernel_size = 2 * inflation_cells + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        inflated_obst = cv2.dilate(obst, kernel)
        inflated_grid = grid.copy()
        inflated_grid[inflated_obst == 1] = 100

        self.grid = inflated_grid
        self.map_received = True

        self.get_logger().info(
            f"收到 /map 并完成障碍膨胀: size=({self.width}x{self.height}), "
            f"res={self.resolution:.3f}, origin=({self.origin_x:.2f},{self.origin_y:.2f}), "
            f"robot_radius={self.robot_radius:.2f}m, inflation_cells={inflation_cells}"
        )

        # 地图更新后，如果已有起点和终点，也可以立刻重新规划
        self.try_plan()

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        # 每次收到新的 initialpose，都更新起点并尝试重新规划
        self.start_pose = msg.pose.pose
        self.get_logger().info(
            f"更新起点: ({self.start_pose.position.x:.2f}, "
            f"{self.start_pose.position.y:.2f})"
        )
        self.try_plan()

    def goal_callback(self, msg: PoseStamped):
        # 每次收到新的 goal，都更新终点并尝试重新规划
        self.goal_pose = msg.pose
        self.get_logger().info(
            f"更新终点: ({self.goal_pose.position.x:.2f}, "
            f"{self.goal_pose.position.y:.2f})"
        )
        self.try_plan()

    # ========== 坐标变换 ==========

    def world_to_grid(self, x: float, y: float):
        """
        世界坐标 (x,y) -> 栅格 (row,col)
        按 OccupancyGrid 约定：origin 在左下角，row 向 +y 增长
        """
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int):
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    # ========== 规划入口 ==========

    def try_plan(self):
        """
        只要：
        - 已收到地图
        - start_pose 和 goal_pose 都不是 None
        就重新执行一次 A*，并发布一条新的 /global_path
        """
        if not self.map_received:
            # 地图还没准备好
            return
        if self.start_pose is None or self.goal_pose is None:
            # 只设置了起点或终点，等待另一端
            return

        sx, sy = self.start_pose.position.x, self.start_pose.position.y
        gx, gy = self.goal_pose.position.x, self.goal_pose.position.y

        sr, sc = self.world_to_grid(sx, sy)
        gr, gc = self.world_to_grid(gx, gy)

        if not self.in_bounds(sr, sc) or not self.in_bounds(gr, gc):
            self.get_logger().warn("起点或终点不在地图范围内，规划失败")
            return

        if not self.is_free(sr, sc):
            self.get_logger().warn("起点在障碍或未知区域，规划失败")
            return
        if not self.is_free(gr, gc):
            self.get_logger().warn("终点在障碍或未知区域，规划失败")
            return

        self.get_logger().info(
            f"开始 A* 规划: start=({sr},{sc}), goal=({gr},{gc})")

        path_rc = self.astar((sr, sc), (gr, gc))

        if path_rc is None:
            self.get_logger().warn("A* 未找到可行路径")
            return

        raw_len = len(path_rc)
        path_rc = self.simplify_path(path_rc)
        self.get_logger().info(
            f"A* 原始路径点数={raw_len}，简化后={len(path_rc)}"
        )

        # ---- 构造 Path 消息 ----
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for r, c in path_rc:
            x, y = self.grid_to_world(r, c)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        # 每次规划成功就发布一条新的 global_path
        self.path_pub.publish(path_msg)
        self.get_logger().info(
            f"已发布 /global_path，点数={len(path_msg.poses)}"
        )

    # ========== A* & 工具函数 ==========

    def in_bounds(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, r, c):
        """
        地图值含义：
          0   = free
          100 = obstacle
          -1  = unknown
        为了更安全，这里 unknown 当障碍。
        调试时你可以改成 `return v != 100` 看效果。
        """
        v = int(self.grid[r, c])
        return v == 0

    def astar(self, start_rc, goal_rc):
        H, W = self.height, self.width
        sr, sc = start_rc
        gr, gc = goal_rc

        def h(r, c):
            return math.hypot(r - gr, c - gc)

        # 8 邻域
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

        open_set = []
        heapq.heappush(open_set, (0.0, (sr, sc)))
        came_from = {}
        g_score = {(sr, sc): 0.0}
        visited = np.zeros((H, W), dtype=bool)

        while open_set:
            _, (r, c) = heapq.heappop(open_set)
            if visited[r, c]:
                continue
            visited[r, c] = True

            if (r, c) == (gr, gc):
                # 回溯路径
                path = [(r, c)]
                while (r, c) in came_from:
                    r, c = came_from[(r, c)]
                    path.append((r, c))
                path.reverse()
                return path

            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not self.in_bounds(nr, nc):
                    continue
                if visited[nr, nc]:
                    continue
                if not self.is_free(nr, nc):
                    continue

                step_cost = math.hypot(dr, dc)
                new_g = g_score[(r, c)] + step_cost

                if (nr, nc) not in g_score or new_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = new_g
                    f = new_g + h(nr, nc)
                    heapq.heappush(open_set, (f, (nr, nc)))
                    came_from[(nr, nc)] = (r, c)

        return None

    # ----- 路径平滑：line-of-sight -----

    def line_is_free(self, r0, c0, r1, c1):
        """
        Bresenham 直线检查：从 (r0,c0) 到 (r1,c1) 之间是否全是 free
        """
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1

        r, c = r0, c0
        if dr >= dc:
            err = dr / 2.0
            while r != r1:
                if not self.is_free(r, c):
                    return False
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
            return self.is_free(r, c)
        else:
            err = dc / 2.0
            while c != c1:
                if not self.is_free(r, c):
                    return False
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
            return self.is_free(r, c)

    def simplify_path(self, path_rc):
        """
        对 A* 得到的 [ (r,c), ... ] 路径做 line-of-sight 简化
        """
        if not path_rc or len(path_rc) <= 2:
            return path_rc

        simplified = [path_rc[0]]
        idx = 0

        while idx < len(path_rc) - 1:
            next_idx = idx + 1
            # 从当前点开始，尽量向后连直线
            for j in range(idx + 1, len(path_rc)):
                r0, c0 = path_rc[idx]
                r1, c1 = path_rc[j]
                if self.line_is_free(r0, c0, r1, c1):
                    next_idx = j
                else:
                    break
            simplified.append(path_rc[next_idx])
            idx = next_idx

        return simplified


def main(args=None):
    rclpy.init(args=args)
    node = AstarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
