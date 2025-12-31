#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import heapq
import os
import re

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
        * 发布两条路径：
            - /global_path       : map 坐标系下的路径（方便 RViz 看）
            - /global_path_utm   : UTM 坐标系下的路径（给 MPC / PP 用）
    """

    def __init__(self):
        super().__init__('astar_planner_node')

        # 机器人“半径”配置（单位米）：车宽的一半 + 安全距离
        self.declare_parameter('robot_radius', 0.5)
        self.robot_radius = float(self.get_parameter('robot_radius').value)

        # 车辆动力学相关配置
        self.declare_parameter('min_turning_radius', 2.0)
        self.declare_parameter('smooth_iterations', 150)
        self.declare_parameter('smooth_weight_data', 0.15)
        self.declare_parameter('smooth_weight_smooth', 0.45)
        self.declare_parameter('curvature_gain', 0.4)
        self.declare_parameter('path_sample_step', 0.2)

        self.min_turning_radius = float(
            self.get_parameter('min_turning_radius').value)
        self.smooth_iterations = int(
            self.get_parameter('smooth_iterations').value)
        self.smooth_weight_data = float(
            self.get_parameter('smooth_weight_data').value)
        self.smooth_weight_smooth = float(
            self.get_parameter('smooth_weight_smooth').value)
        self.curvature_gain = float(
            self.get_parameter('curvature_gain').value)
        self.path_sample_step = float(
            self.get_parameter('path_sample_step').value)

        # === 新增：读取 map.yaml，用于构造 map -> UTM 的线性变换 ===
        #   map_yaml_path 需要在 launch 里传进来，例如：
        #   <param name="map_yaml_path" value="/home/.../map.yaml"/>
        self.declare_parameter('map_yaml_path', '/home/wudi/slam/get_pc_from_db3/map.yaml')
        self.map_yaml_path = self.get_parameter('map_yaml_path').get_parameter_value().string_value

        # PNG 左下角对应的 UTM 坐标（从 map.yaml 注释中解析）
        self.png_origin_e = None  # E_png
        self.png_origin_n = None  # N_png

        # UTM 原点（建图时 lat0,lon0 对应的 UTM）
        # map 坐标 (x_map, y_map) -> UTM: E = utm_origin_e + x_map, N = utm_origin_n + y_map
        self.utm_origin_e = None
        self.utm_origin_n = None

        if self.map_yaml_path:
            self.load_png_origin_utm(self.map_yaml_path)
        else:
            self.get_logger().warn(
                "未设置 map_yaml_path 参数，将无法发布 UTM 坐标的路径 (/global_path_utm)"
            )

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
        # map 坐标系的全局路径（主要给 RViz 看）
        self.path_pub = self.create_publisher(Path, 'global_path', path_qos)
        # UTM 坐标系的全局路径（给 MPC / PP 使用）
        self.path_pub_utm = self.create_publisher(Path, 'global_path_utm', path_qos)

        self.get_logger().info(
            "AstarPlannerNode 启动，等待 /map、/initialpose、/goal_pose ..."
        )

    # ========== 解析 map.yaml 中的 PNG origin UTM ==========

    def load_png_origin_utm(self, yaml_path: str):
        """
        从 map.yaml 中解析
            # PNG origin UTM: E=..., N=...
        这行注释，得到 PNG 左下角对应的 UTM 坐标。
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
                "将无法发布 UTM 路径。"
            )
        except Exception as e:
            self.get_logger().error(f"解析 map.yaml 失败: {e}")

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

        # === 一旦有了地图的 origin_xy，并且之前解析出了 PNG origin UTM，就可以算 UTM 原点 ===
        if self.png_origin_e is not None and self.png_origin_n is not None:
            # E_png = E0 + origin_x  =>  E0 = E_png - origin_x
            self.utm_origin_e = self.png_origin_e - self.origin_x
            self.utm_origin_n = self.png_origin_n - self.origin_y
            self.get_logger().info(
                f"计算得到 utm_origin: E0={self.utm_origin_e:.3f}, N0={self.utm_origin_n:.3f} "
                f"(map (0,0) 对应的 UTM 坐标)"
            )
        else:
            self.get_logger().warn(
                "尚未获取 PNG origin UTM，因此无法计算 utm_origin，将只发布 map 坐标的路径。"
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
        """
        栅格 (row,col) -> 世界坐标 (map 坐标系下的 x,y)
        """
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    # ========== 规划入口 ==========

    def try_plan(self):
        """
        只要：
        - 已收到地图
        - start_pose 和 goal_pose 都不是 None
        就重新执行一次 A*，并发布 /global_path（map） 和 /global_path_utm（UTM）
        """
        if not self.map_received:
            return
        if self.start_pose is None or self.goal_pose is None:
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

        # 将栅格坐标路径转换为世界坐标，并进行平滑（map 坐标系）
        world_path = [self.grid_to_world(r, c) for r, c in path_rc]
        world_path = self.vehicle_friendly_smoothing(world_path)

        # 计算每个路径点的朝向（yaw），便于下游控制
        yaws = self.estimate_yaws(world_path)

        # ---- 构造 map 坐标系的 Path 消息 ----
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id  # 一般是 "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for idx, (x, y) in enumerate(world_path):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            qw, qx, qy, qz = self.yaw_to_quaternion(yaws[idx])
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path_msg.poses.append(ps)

        self.path_pub.publish(path_msg)
        self.get_logger().info(
            f"已发布 /global_path (map 坐标)，点数={len(path_msg.poses)}"
        )

        # ---- 构造 UTM 坐标系的 Path 消息（给 MPC / PP 使用）----
        if self.utm_origin_e is not None and self.utm_origin_n is not None:
            path_msg_utm = Path()
            path_msg_utm.header.frame_id = "utm"
            path_msg_utm.header.stamp = self.get_clock().now().to_msg()

            for idx, (x_map, y_map) in enumerate(world_path):
                # map -> UTM: E = E0 + x_map, N = N0 + y_map
                E = self.utm_origin_e + x_map
                N = self.utm_origin_n + y_map

                ps = PoseStamped()
                ps.header = path_msg_utm.header
                ps.pose.position.x = E
                ps.pose.position.y = N
                ps.pose.position.z = 0.0
                qw, qx, qy, qz = self.yaw_to_quaternion(yaws[idx])
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw
                path_msg_utm.poses.append(ps)

            self.path_pub_utm.publish(path_msg_utm)
            self.get_logger().info(
                f"已发布 /global_path_utm (UTM 坐标)，点数={len(path_msg_utm.poses)}"
            )
        else:
            self.get_logger().warn(
                "未计算出 utm_origin，无法发布 /global_path_utm，如需 UTM 路径请确保 map.yaml 中有 "
                "'# PNG origin UTM: E=..., N=...' 注释并设置 map_yaml_path 参数。"
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

    # ----- 平滑 & 动力学约束 -----

    def vehicle_friendly_smoothing(self, world_path):
        """
        先利用梯度平滑减少折线，再按最小转弯半径约束对拐点进行调整。
        """
        if len(world_path) <= 2:
            return world_path

        points = np.array(world_path, dtype=np.float64)
        smoothed = points.copy()
        max_curvature = 1.0 / max(self.min_turning_radius, 1e-3)

        for _ in range(max(self.smooth_iterations, 1)):
            max_delta = 0.0
            for i in range(1, len(points) - 1):
                prev_pt = smoothed[i - 1]
                next_pt = smoothed[i + 1]
                orig_pt = points[i]
                current = smoothed[i]

                data_term = self.smooth_weight_data * (orig_pt - current)
                smooth_term = self.smooth_weight_smooth * \
                    (prev_pt + next_pt - 2.0 * current)
                current = current + data_term + smooth_term

                radius, center = self.turning_radius(prev_pt, current, next_pt)
                if radius is not None and radius > 1e-6:
                    curvature = 1.0 / radius
                    if curvature > max_curvature and center is not None:
                        direction = current - center
                        norm = np.linalg.norm(direction)
                        if norm > 1e-6:
                            direction = direction / norm
                            diff = (curvature - max_curvature) / \
                                max_curvature
                            current = current + \
                                self.curvature_gain * diff * direction

                # 保证新点及与相邻点连线均处于自由区
                if self.is_world_free_point(current) and \
                        self.segment_collision_free(prev_pt, current) and \
                        self.segment_collision_free(current, next_pt):
                    max_delta = max(
                        max_delta,
                        float(np.linalg.norm(smoothed[i] - current))
                    )
                    smoothed[i] = current

            if max_delta < 1e-4:
                break

        smoothed_list = smoothed.tolist()
        return self.resample_path(smoothed_list, self.path_sample_step)

    def turning_radius(self, p0, p1, p2):
        """
        返回三点确定圆的半径和圆心。如果三点共线则返回 (None, None)。
        """
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        d = 2 * (x0 * (y1 - y2) +
                 x1 * (y2 - y0) +
                 x2 * (y0 - y1))
        if abs(d) < 1e-9:
            return None, None

        ux = ((x0**2 + y0**2) * (y1 - y2) +
              (x1**2 + y1**2) * (y2 - y0) +
              (x2**2 + y2**2) * (y0 - y1)) / d
        uy = ((x0**2 + y0**2) * (x2 - x1) +
              (x1**2 + y1**2) * (x0 - x2) +
              (x2**2 + y2**2) * (x1 - x0)) / d
        center = np.array([ux, uy])
        radius = float(np.linalg.norm(center - np.array(p0)))
        return radius, center

    def resample_path(self, path, step):
        """
        以固定步长重新采样轨迹，方便控制器追踪。
        """
        if step <= 1e-3 or len(path) < 2:
            return path

        resampled = [path[0]]
        accumulated = 0.0
        prev = np.array(path[0], dtype=np.float64)

        for i in range(1, len(path)):
            segment_end = np.array(path[i], dtype=np.float64)
            seg_vec = segment_end - prev
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-6:
                prev = segment_end
                continue
            direction = seg_vec / seg_len
            remaining = seg_len

            while accumulated + remaining >= step - 1e-9:
                need = step - accumulated
                prev = prev + direction * need
                resampled.append(prev.tolist())
                remaining -= need
                accumulated = 0.0
                if remaining < 1e-6:
                    break

            accumulated += remaining
            prev = segment_end

        if np.linalg.norm(np.array(resampled[-1]) - np.array(path[-1])) > 1e-3:
            resampled.append(path[-1])
        else:
            resampled[-1] = path[-1]
        return resampled

    def is_world_free_point(self, point):
        """
        检查世界坐标点是否在自由区内。
        """
        x, y = point
        r, c = self.world_to_grid(x, y)
        if not self.in_bounds(r, c):
            return False
        return self.is_free(r, c)

    def segment_collision_free(self, start, end):
        """
        对世界坐标线段按地图分辨率采样，检查是否穿过障碍。
        """
        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)
        seg = end - start
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            return self.is_world_free_point(start)

        direction = seg / length
        steps = max(int(length / max(self.resolution * 0.5, 1e-3)), 1)
        for i in range(steps + 1):
            point = start + direction * (length * i / steps)
            if not self.is_world_free_point(point):
                return False
        return True

    def estimate_yaws(self, world_path):
        """
        根据连续点计算朝向（yaw）。
        """
        if len(world_path) == 1:
            return [0.0]

        yaws = []
        for i in range(len(world_path)):
            if i == len(world_path) - 1:
                dx = world_path[i][0] - world_path[i - 1][0]
                dy = world_path[i][1] - world_path[i - 1][1]
            else:
                dx = world_path[i + 1][0] - world_path[i][0]
                dy = world_path[i + 1][1] - world_path[i][1]
            yaws.append(math.atan2(dy, dx))
        return yaws

    def yaw_to_quaternion(self, yaw):
        """
        将偏航角转换为四元数 (w,x,y,z)。
        """
        half = yaw * 0.5
        return (
            math.cos(half),
            0.0,
            0.0,
            math.sin(half)
        )


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
