#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UTM-based local planner with Frenet-lattice sampling.

订阅:
- /global_path_utm (nav_msgs/Path, frame_id="utm")
- /pose_utm       (geometry_msgs/PoseStamped, frame_id="utm")
- /fast_obstacles (std_msgs/String, JSON, 障碍物在自车坐标系下)
- /odom           (nav_msgs/Odometry，可选，仅用于获取当前速度)

发布:
- /local_path         (nav_msgs/Path)  # 当前选中的局部轨迹
- /update_global_path (nav_msgs/Path)  # 局部轨迹 + 剩余全局路径

算法:
- 使用 Frenet 坐标 (s, l) 进行 lattice 轨迹采样:
    * 纵向 s(t): quartic 多项式 (s0, s0_v, s0_a → s1_v, 0)
    * 横向 l(t): quintic 多项式 (l0, l0_v, l0_a → l1, 0, 0)
- 多个 {t1, s1_v, l1} 组合形成候选轨迹集合
- 在参考全局路径上做 SL → XY 转换
- 对每条轨迹计算:
    * jerk (横向 + 纵向)
    * 末端速度偏差 |v(T) - v_target|
    * 时间 t1
    * 横向偏移 |l(T)|
    * 碰撞 (基于 fast_obstacles 转到 UTM 后的距离阈值)
- 选择 cost 最低且满足速度/加速度/曲率约束的轨迹
- 将该局部轨迹发布为 /local_path，并与剩余全局路径拼接为 /update_global_path
"""

import math
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


# ===================== Frenet 参考路径 =====================

class FrenetReferencePath:
    """
    使用离散全局路径点构造的简易 Frenet 参考线:
    - s: 弧长
    - x(s), y(s) 线性插值
    - yaw(s) 使用段 yaw
    """

    def __init__(self, xy_points: List[Tuple[float, float]]):
        self.x = [p[0] for p in xy_points]
        self.y = [p[1] for p in xy_points]

        if len(self.x) < 2:
            self.s = [0.0]
            self.yaw = [0.0]
            self.length = 0.0
            return

        # 弧长
        self.s = [0.0]
        for i in range(1, len(self.x)):
            dx = self.x[i] - self.x[i - 1]
            dy = self.y[i] - self.y[i - 1]
            self.s.append(self.s[-1] + math.hypot(dx, dy))
        self.length = self.s[-1]

        # yaw
        self.yaw = []
        for i in range(len(self.x) - 1):
            dx = self.x[i + 1] - self.x[i]
            dy = self.y[i + 1] - self.y[i]
            self.yaw.append(math.atan2(dy, dx))
        self.yaw.append(self.yaw[-1])

    def calc_position(self, s: float) -> Tuple[Optional[float], Optional[float]]:
        if s < 0.0 or s > self.length or len(self.s) < 2:
            return None, None

        # 找到 s 所在的段
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i + 1]:
                ratio = (s - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-6)
                x = self.x[i] + ratio * (self.x[i + 1] - self.x[i])
                y = self.y[i] + ratio * (self.y[i + 1] - self.y[i])
                return x, y

        return self.x[-1], self.y[-1]

    def calc_yaw(self, s: float) -> float:
        if s <= 0.0:
            return self.yaw[0]
        if s >= self.length:
            return self.yaw[-1]

        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i + 1]:
                return self.yaw[i]
        return self.yaw[-1]


# ===================== 多项式轨迹 (参考 lattice_planner) =====================

class QuarticPolynomial:
    """ s(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4
    条件: s(0)=s0, s'(0)=v0, s''(0)=a0
          s'(T)=v1, s''(T)=a1
    """

    def __init__(self, x0, v0, a0, v1, a1, T):
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T

        A = np.array([
            [3 * T2, 4 * T3],
            [6 * T, 12 * T2]
        ])
        b = np.array([
            v1 - self.a1 - 2 * self.a2 * T,
            a1 - 2 * self.a2
        ])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_xt(self, t):
        return (self.a0 + self.a1 * t + self.a2 * t ** 2 +
                self.a3 * t ** 3 + self.a4 * t ** 4)

    def calc_dxt(self, t):
        return (self.a1 + 2 * self.a2 * t +
                3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3)

    def calc_ddxt(self, t):
        return (2 * self.a2 + 6 * self.a3 * t +
                12 * self.a4 * t ** 2)

    def calc_dddxt(self, t):
        return 6 * self.a3 + 24 * self.a4 * t


class QuinticPolynomial:
    """ l(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
    条件: l(0)=l0, l'(0)=lv0, l''(0)=la0
          l(T)=l1, l'(T)=lv1, l''(T)=la1
    """

    def __init__(self, x0, v0, a0, x1, v1, a1, T):
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        A = np.array([
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3]
        ])
        b = np.array([
            x1 - self.a0 - self.a1 * T - self.a2 * T2,
            v1 - self.a1 - 2 * self.a2 * T,
            a1 - 2 * self.a2
        ])
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_xt(self, t):
        return (self.a0 + self.a1 * t + self.a2 * t ** 2 +
                self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5)

    def calc_dxt(self, t):
        return (self.a1 + 2 * self.a2 * t +
                3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4)

    def calc_ddxt(self, t):
        return (2 * self.a2 + 6 * self.a3 * t +
                12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3)

    def calc_dddxt(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2


@dataclass
class LatticePath:
    t: List[float] = field(default_factory=list)
    l: List[float] = field(default_factory=list)
    l_v: List[float] = field(default_factory=list)
    l_a: List[float] = field(default_factory=list)
    l_jerk: List[float] = field(default_factory=list)
    s: List[float] = field(default_factory=list)
    s_v: List[float] = field(default_factory=list)
    s_a: List[float] = field(default_factory=list)
    s_jerk: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    ds: List[float] = field(default_factory=list)
    curv: List[float] = field(default_factory=list)
    cost: float = 0.0


# ===================== Lattice 参数 (参考 C 类) =====================

class LatticeConfig:
    # 速度/曲率约束
    MAX_SPEED = 50.0 / 3.6
    MAX_ACCEL = 8.0
    MAX_CURVATURE = 2.5

    # 横向采样宽度
    ROAD_WIDTH = 6.0          # [m] 以中心线为 0，两边各 6m
    ROAD_SAMPLE_STEP = 1.0    # [m]

    # 时间离散
    T_STEP = 0.15
    MIN_T = 4.0
    MAX_T = 5.0

    # 目标速度
    TARGET_SPEED = 30.0 / 3.6
    SPEED_SAMPLE_STEP = 5.0 / 3.6

    # Cost 权重
    K_JERK = 0.1
    K_TIME = 0.1
    K_V_DIFF = 1.0
    K_OFFSET = 1.5
    K_COLLISION = 500.0

    # 车辆尺寸（用于曲率约束，碰撞简单用圆）
    WB = 3.5
    RF = 4.5
    RB = 1.0
    W = 3.0


# ===================== 工具函数 =====================

def calc_yaw_curv(x: List[float], y: List[float]):
    yaw = []
    ds = []
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        ds_i = math.hypot(dx, dy)
        ds.append(max(ds_i, 1e-6))
        yaw.append(math.atan2(dy, dx))

    if not yaw:
        return None, None, None

    yaw.append(yaw[-1])
    ds.append(ds[-1])

    curv = []
    for i in range(len(yaw) - 1):
        curv.append((yaw[i + 1] - yaw[i]) / ds[i])

    return yaw, curv, ds


def sl_to_xy(s_list: List[float], l_list: List[float], ref_path: FrenetReferencePath):
    xs, ys = [], []
    for s, l in zip(s_list, l_list):
        x_ref, y_ref = ref_path.calc_position(s)
        if x_ref is None:
            break
        yaw_ref = ref_path.calc_yaw(s)
        # 法向量 (左侧)
        nx = math.cos(yaw_ref + math.pi / 2.0)
        ny = math.sin(yaw_ref + math.pi / 2.0)
        xs.append(x_ref + l * nx)
        ys.append(y_ref + l * ny)
    return xs, ys


def is_path_collision(path: LatticePath,
                      obstacles: List[Tuple[float, float]],
                      robot_radius: float) -> bool:
    if not obstacles:
        return False

    # 下采样一些点做碰撞检测
    step = max(len(path.x) // 20, 1)
    for i in range(0, len(path.x), step):
        px = path.x[i]
        py = path.y[i]
        for ox, oy in obstacles:
            if math.hypot(px - ox, py - oy) <= robot_radius:
                return True
    return False


def verify_path(path: LatticePath) -> bool:
    cfg = LatticeConfig
    if any(v > cfg.MAX_SPEED + 1e-3 for v in path.s_v):
        return False
    if any(abs(a) > cfg.MAX_ACCEL + 1e-3 for a in path.s_a):
        return False
    if any(abs(k) > cfg.MAX_CURVATURE + 1e-3 for k in path.curv):
        return False
    return True


def extract_optimal_path(candidates: Dict[LatticePath, float]) -> Optional[LatticePath]:
    if not candidates:
        return None
    # 直接按 cost 排序，找第一个满足约束的
    for path in sorted(candidates.keys(), key=lambda p: candidates[p]):
        if verify_path(path):
            return path
    return None


# ===================== ROS2 节点：Lattice 局部规划 =====================

class LatticeLocalPlannerNode(Node):
    def __init__(self):
        super().__init__("lattice_local_planner")

        qos_transient = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        qos_best_effort = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # 订阅
        self.global_path_sub = self.create_subscription(
            Path, "/global_path_utm", self.global_path_callback, qos_transient
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, "/pose_utm", self.pose_callback, qos_best_effort
        )
        self.fast_obstacle_sub = self.create_subscription(
            String, "/fast_obstacles", self.fast_obstacle_callback, qos_best_effort
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_best_effort
        )

        # 发布
        self.local_path_pub = self.create_publisher(
            Path, "/local_path", qos_best_effort
        )
        self.updated_global_path_pub = self.create_publisher(
            Path, "/update_global_path", qos_best_effort
        )

        # 内部状态
        self.global_path_msg: Optional[Path] = None
        self.global_points: List[Tuple[float, float]] = []
        self.ref_path: Optional[FrenetReferencePath] = None

        self.current_pose: Optional[PoseStamped] = None
        self.current_speed: float = 0.0  # [m/s]

        self.relative_obstacles: List[Tuple[float, float]] = []
        self.global_obstacles: List[Tuple[float, float]] = []

        self.global_frame_id = "utm"
        self.robot_radius = 2.0  # 用于碰撞检测的圆半径

        self.closest_path_idx = 0

        # 定时器
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

    # ---------- 回调 ----------

    def global_path_callback(self, msg: Path):
        if not msg.poses:
            return
        self.global_path_msg = msg
        self.global_frame_id = msg.header.frame_id or "utm"
        self.global_points = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]
        self.ref_path = FrenetReferencePath(self.global_points)
        self.closest_path_idx = 0
        self.get_logger().info(
            f"Received global_path_utm with {len(self.global_points)} points."
        )

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    def fast_obstacle_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("fast_obstacles JSON 解析失败")
            return

        obs_list = data.get("obstacles", [])
        rel = []
        for o in obs_list:
            try:
                x = float(o.get("x", 0.0))
                y = float(o.get("y", 0.0))
                rel.append((x, y))
            except Exception:
                continue
        self.relative_obstacles = rel
        self.update_obstacle_world_coordinates()

    # ---------- 坐标变换 / Frenet 相关 ----------

    def update_obstacle_world_coordinates(self):
        if not self.relative_obstacles or self.current_pose is None:
            self.global_obstacles = []
            return

        yaw = self.yaw_from_pose(self.current_pose)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        base_x = self.current_pose.pose.position.x
        base_y = self.current_pose.pose.position.y

        transformed = []
        for rx, ry in self.relative_obstacles:
            gx = base_x + rx * cos_yaw - ry * sin_yaw
            gy = base_y + rx * sin_yaw + ry * cos_yaw
            transformed.append((gx, gy))

        self.global_obstacles = transformed

    @staticmethod
    def yaw_from_pose(pose_stamped: PoseStamped) -> float:
        q = pose_stamped.pose.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def cartesian_to_frenet(self) -> Optional[Tuple[float, float]]:
        """
        将当前 UTM 坐标转换为 Frenet (s, l).
        s: 沿参考线的弧长
        l: 相对参考线横向偏移
        """
        if self.ref_path is None or self.current_pose is None:
            return None

        px = self.current_pose.pose.position.x
        py = self.current_pose.pose.position.y

        # 简单最近点搜索
        best_idx = 0
        best_dist2 = float("inf")
        for i, (gx, gy) in enumerate(self.global_points):
            d2 = (gx - px) ** 2 + (gy - py) ** 2
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i

        self.closest_path_idx = best_idx
        s_base = self.ref_path.s[best_idx] if best_idx < len(self.ref_path.s) else 0.0
        x_ref = self.ref_path.x[best_idx]
        y_ref = self.ref_path.y[best_idx]
        yaw_ref = self.ref_path.yaw[best_idx]

        dx = px - x_ref
        dy = py - y_ref

        # 参考线切向/法向
        tx = math.cos(yaw_ref)
        ty = math.sin(yaw_ref)
        nx = -math.sin(yaw_ref)
        ny = math.cos(yaw_ref)

        s_offset = dx * tx + dy * ty
        l = dx * nx + dy * ny

        s = s_base + s_offset
        s = max(0.0, min(s, self.ref_path.length))

        return s, l

    # ---------- Lattice 采样与评分 ----------

    def sampling_paths_for_cruising(
        self,
        l0: float,
        l0_v: float,
        l0_a: float,
        s0: float,
        s0_v: float,
        s0_a: float,
    ) -> Dict[LatticePath, float]:
        cfg = LatticeConfig
        paths: Dict[LatticePath, float] = {}

        if self.ref_path is None:
            return paths

        # 纵向目标速度采样
        for s1_v in np.arange(
            cfg.TARGET_SPEED * 0.6,
            cfg.TARGET_SPEED * 1.4,
            cfg.TARGET_SPEED * 0.2,
        ):
            # 到达该目标速度的时间采样
            for t1 in np.arange(cfg.MIN_T, cfg.MAX_T + 1e-6, 0.2):
                path_pre = LatticePath()
                lon_poly = QuarticPolynomial(s0, s0_v, s0_a, s1_v, 0.0, t1)

                t_list = list(np.arange(0.0, t1, cfg.T_STEP))
                path_pre.t = t_list
                path_pre.s = [lon_poly.calc_xt(t) for t in t_list]
                path_pre.s_v = [lon_poly.calc_dxt(t) for t in t_list]
                path_pre.s_a = [lon_poly.calc_ddxt(t) for t in t_list]
                path_pre.s_jerk = [lon_poly.calc_dddxt(t) for t in t_list]

                # 横向目标偏移采样
                for l1 in np.arange(
                    -cfg.ROAD_WIDTH, cfg.ROAD_WIDTH + 1e-6, cfg.ROAD_SAMPLE_STEP
                ):
                    lat_poly = QuinticPolynomial(l0, l0_v, l0_a, l1, 0.0, 0.0, t1)
                    path = LatticePath()
                    path.t = path_pre.t
                    path.s = path_pre.s
                    path.s_v = path_pre.s_v
                    path.s_a = path_pre.s_a
                    path.s_jerk = path_pre.s_jerk

                    path.l = [lat_poly.calc_xt(t) for t in path_pre.t]
                    path.l_v = [lat_poly.calc_dxt(t) for t in path_pre.t]
                    path.l_a = [lat_poly.calc_ddxt(t) for t in path_pre.t]
                    path.l_jerk = [lat_poly.calc_dddxt(t) for t in path_pre.t]

                    # SL → XY
                    path.x, path.y = sl_to_xy(path.s, path.l, self.ref_path)
                    if len(path.x) < 2:
                        continue

                    path.yaw, path.curv, path.ds = calc_yaw_curv(path.x, path.y)
                    if path.yaw is None:
                        continue

                    # cost
                    l_jerk_sum = float(np.sum(np.abs(path.l_jerk)))
                    s_jerk_sum = float(np.sum(np.abs(path.s_jerk)))
                    v_diff = abs(cfg.TARGET_SPEED - path.s_v[-1])

                    collision = is_path_collision(
                        path, self.global_obstacles, self.robot_radius
                    )

                    path.cost = (
                        cfg.K_JERK * (l_jerk_sum + s_jerk_sum)
                        + cfg.K_V_DIFF * v_diff
                        + cfg.K_TIME * t1 * 2.0
                        + cfg.K_OFFSET * abs(path.l[-1])
                        + cfg.K_COLLISION * (1.0 if collision else 0.0)
                    )

                    paths[path] = path.cost

        return paths

    # ---------- 控制循环 ----------

    def control_loop(self):
        if (
            self.global_path_msg is None
            or self.ref_path is None
            or self.current_pose is None
        ):
            return

        # 到达全局终点则不再规划
        if self.reached_goal():
            return

        frenet = self.cartesian_to_frenet()
        if frenet is None:
            return

        s0, l0 = frenet
        # 简化：横向速度/加速度取 0，纵向加速度取 0
        l0_v = 0.0
        l0_a = 0.0
        s0_v = max(self.current_speed, 0.1)
        s0_a = 0.0

        # 采样 lattice 轨迹
        candidates = self.sampling_paths_for_cruising(l0, l0_v, l0_a, s0, s0_v, s0_a)
        best = extract_optimal_path(candidates)

        if best is None or len(best.x) < 2:
            self.get_logger().warn("Lattice: 无可行轨迹")
            return

        # 将 best 转为 ROS Path
        local_path_msg = Path()
        local_path_msg.header.frame_id = self.global_frame_id
        local_path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in zip(best.x, best.y):
            ps = PoseStamped()
            ps.header = local_path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            local_path_msg.poses.append(ps)

        self.local_path_pub.publish(local_path_msg)

        # 拼接 updated_global_path: 局部轨迹 + 剩余全局
        self.publish_updated_global_path(best)

    def reached_goal(self) -> bool:
        if self.global_path_msg is None or self.current_pose is None:
            return False
        goal_pose = self.global_path_msg.poses[-1].pose
        gx = goal_pose.position.x
        gy = goal_pose.position.y
        px = self.current_pose.pose.position.x
        py = self.current_pose.pose.position.y
        dist = math.hypot(gx - px, gy - py)
        return dist < 2.0

    def publish_updated_global_path(self, best: LatticePath):
        if self.global_path_msg is None:
            return

        path_msg = Path()
        path_msg.header.frame_id = self.global_frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # 1) 先放局部 lattice 轨迹
        for x, y in zip(best.x, best.y):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            path_msg.poses.append(ps)

        # 2) 找到 lattice 轨迹末端在全局路径上的最近点索引
        end_x = best.x[-1]
        end_y = best.y[-1]

        best_idx = 0
        best_d2 = float("inf")
        for i, (gx, gy) in enumerate(self.global_points):
            d2 = (gx - end_x) ** 2 + (gy - end_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        # 3) 拼接剩余全局路径
        for i in range(best_idx, len(self.global_path_msg.poses)):
            pose = self.global_path_msg.poses[i]
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose = pose.pose
            path_msg.poses.append(ps)

        self.updated_global_path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LatticeLocalPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
