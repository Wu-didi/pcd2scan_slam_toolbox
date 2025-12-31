#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UTM-based local planner with fast obstacle detection.

订阅:
- /global_path_utm (nav_msgs/Path, frame_id="utm")
- /pose_utm       (geometry_msgs/PoseStamped, frame_id="utm")
- /fast_obstacles (std_msgs/String, JSON, 障碍物在自车坐标系下)
- /odom           (nav_msgs/Odometry，可选，仅用于获取当前速度)

发布:
- /local_path         (nav_msgs/Path)          # 当前选中的局部轨迹
- /update_global_path (nav_msgs/Path)          # 局部轨迹 + 剩余全局路径
- /cmd_vel            (geometry_msgs/Twist)    # 速度控制指令

算法:
- 基于 Dynamic Window Approach (DWA) 对 (v, w) 采样
- 使用 global_path_utm 作为参考路径，选取前视目标点
- 将 fast_obstacles 中自车系障碍物转换到 UTM 系，进行避障
"""

import math
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String


def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    while angle >= math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_pose(pose: Pose) -> float:
    """从 Pose 的四元数中提取 Z 轴 yaw。"""
    q = pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def pose_copy(src: Pose) -> Pose:
    dst = Pose()
    dst.position.x = src.position.x
    dst.position.y = src.position.y
    dst.position.z = src.position.z
    dst.orientation.x = src.orientation.x
    dst.orientation.y = src.orientation.y
    dst.orientation.z = src.orientation.z
    dst.orientation.w = src.orientation.w
    return dst


@dataclass
class TrajectoryCandidate:
    linear_velocity: float
    angular_velocity: float
    poses: List[PoseStamped]
    min_clearance: float
    first_collision_time: Optional[float]
    heading_error: float
    path_error: float
    score: float
    target_index: Optional[int]


class UtmLocalPlannerNode(Node):
    def __init__(self):
        super().__init__('local_path_planner_node')

        # --------- 参数 ---------
        # 话题名
        self.declare_parameter('global_path_topic', 'global_path_utm')
        self.declare_parameter('pose_topic', 'pose_utm')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('local_path_topic', 'local_path')
        self.declare_parameter('update_global_path_topic', 'update_global_path')
        self.declare_parameter('obstacle_topic', 'fast_obstacles')
        self.declare_parameter('velocity_topic', 'odom')
        self.declare_parameter('use_odom_velocity', True)

        # 路径 & 控制相关
        self.declare_parameter('lookahead_distance', 10.0)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('robot_radius', 0.7)
        self.declare_parameter('control_frequency', 10.0)

        # 预测 & DWA 采样
        self.declare_parameter('predict_time', 10.0)
        self.declare_parameter('predict_dt', 1)
        self.declare_parameter('velocity_samples', 5)
        self.declare_parameter('yaw_samples', 12)

        # 速度/加速度约束
        self.declare_parameter('max_speed', 7.0)
        self.declare_parameter('min_speed', 0.0)
        self.declare_parameter('max_yaw_rate', 1.2)
        self.declare_parameter('max_acceleration', 1.5)
        self.declare_parameter('max_deceleration', 2.0)
        self.declare_parameter('max_yaw_acceleration', 2.5)

        # 代价权重
        self.declare_parameter('path_follow_weight', 1.2)
        self.declare_parameter('heading_weight', 1.4)
        self.declare_parameter('velocity_weight', 0.4)
        self.declare_parameter('clearance_weight', 2.4)
        self.declare_parameter('ttc_weight', 1.0)

        # lattice 偏置
        self.declare_parameter('lattice_yaw_bias', [-0.8, -0.4, 0.0, 0.4, 0.8])

        # --------- 读取参数 ---------
        self.global_path_topic = self.get_parameter('global_path_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.local_path_topic = self.get_parameter('local_path_topic').value
        self.update_global_path_topic = self.get_parameter('update_global_path_topic').value
        self.obstacle_topic = self.get_parameter('obstacle_topic').value
        self.velocity_topic = self.get_parameter('velocity_topic').value
        self.use_odom_velocity = bool(self.get_parameter('use_odom_velocity').value)

        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.control_frequency = max(1.0, float(self.get_parameter('control_frequency').value))

        self.predict_time = max(0.2, float(self.get_parameter('predict_time').value))
        self.predict_dt = max(0.02, float(self.get_parameter('predict_dt').value))
        self.velocity_samples = max(3, int(self.get_parameter('velocity_samples').value))
        self.yaw_samples = max(3, int(self.get_parameter('yaw_samples').value))

        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.max_acceleration = float(self.get_parameter('max_acceleration').value)
        self.max_deceleration = float(self.get_parameter('max_deceleration').value)
        self.max_yaw_acceleration = float(self.get_parameter('max_yaw_acceleration').value)

        self.path_follow_weight = float(self.get_parameter('path_follow_weight').value)
        self.heading_weight = float(self.get_parameter('heading_weight').value)
        self.velocity_weight = float(self.get_parameter('velocity_weight').value)
        self.clearance_weight = float(self.get_parameter('clearance_weight').value)
        self.ttc_weight = float(self.get_parameter('ttc_weight').value)

        lattice_bias_param = self.get_parameter('lattice_yaw_bias').value
        self.lattice_yaw_bias = [float(val) for val in (lattice_bias_param or [])]

        self.control_period = 1.0 / self.control_frequency

        # --------- 状态 ---------
        self.global_path_msg: Optional[Path] = None
        self.global_path_points: List[Tuple[float, float]] = []
        self.global_frame_id: str = 'utm'
        self.closest_path_idx: int = 0

        self.current_pose: Optional[Pose] = None
        self.pose_frame_id: str = 'utm'

        self.current_linear_vel: float = 0.0
        self.current_angular_vel: float = 0.0
        self.have_velocity: bool = False

        # fast_obstacles
        self.relative_obstacles: List[Tuple[float, float]] = []   # 自车系 (x,y)
        self.global_obstacles: List[Tuple[float, float]] = []     # UTM 系 (E,N)
        self.obstacle_frame_id: Optional[str] = None

        self.warned_frame_mismatch = False

        # --------- 订阅 /global_path_utm ---------
        global_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.global_path_sub = self.create_subscription(
            Path,
            self.global_path_topic,
            self.global_path_callback,
            global_qos
        )

        # --------- 订阅 pose_utm ---------
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10
        )

        # --------- 订阅 odom（可选，仅速度） ---------
        if self.use_odom_velocity:
            self.odom_sub = self.create_subscription(
                Odometry,
                self.velocity_topic,
                self.odom_callback,
                10
            )
        else:
            self.odom_sub = None

        # --------- 订阅 fast_obstacles ---------
        self.obstacle_sub = self.create_subscription(
            String,
            self.obstacle_topic,
            self.fast_obstacle_callback,
            10
        )

        # --------- 发布 ---------
        self.local_path_pub = self.create_publisher(Path, self.local_path_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.updated_global_path_pub = self.create_publisher(
            Path, self.update_global_path_topic, 10
        )

        # --------- 控制循环 ---------
        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            "UTM Local Planner 启动: "
            f"global_path_topic={self.global_path_topic}, "
            f"pose_topic={self.pose_topic}, "
            f"obstacle_topic={self.obstacle_topic}, "
            f"velocity_topic={self.velocity_topic}, "
            f"use_odom_velocity={self.use_odom_velocity}"
        )

    # ================== 订阅回调 ==================

    def global_path_callback(self, msg: Path):
        self.global_path_msg = msg
        self.global_frame_id = msg.header.frame_id or 'utm'
        self.global_path_points = [
            (pose.pose.position.x, pose.pose.position.y) for pose in msg.poses
        ]
        self.closest_path_idx = 0
        self.get_logger().info(
            f"收到 global_path: {len(self.global_path_points)} points, frame={self.global_frame_id}"
        )

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = pose_copy(msg.pose)
        self.pose_frame_id = msg.header.frame_id or 'utm'
        self.update_obstacle_world_coordinates()

    def odom_callback(self, msg: Odometry):
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z
        self.have_velocity = True

    def fast_obstacle_callback(self, msg: String):
        """
        接收 fast_lidar_obstacle_detector 的 JSON:
        {
          "stamp": {...},
          "frame_id": "base_link" / "rslidar" / ...,
          "has_obstacle": true/false,
          "obstacles": [
            {"x":..., "y":..., "z":..., "count": N},
            ...
          ]
        }
        这里假设 x,y 在自车坐标系下。
        """
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"解析 fast_obstacles JSON 失败: {e}")
            return

        self.obstacle_frame_id = data.get("frame_id", self.obstacle_frame_id)

        obs_list = data.get("obstacles", [])
        rel: List[Tuple[float, float]] = []
        for o in obs_list:
            try:
                x = float(o.get("x", 0.0))
                y = float(o.get("y", 0.0))
            except Exception:
                continue
            rel.append((x, y))
        self.relative_obstacles = rel

        # 有了新障碍物，在当前位姿下立刻更新 UTM 坐标
        self.update_obstacle_world_coordinates()

    # ================== 核心控制循环 ==================

    def control_loop(self):
        if not self.global_path_points or self.current_pose is None:
            return

        # 检查坐标系是否一致
        if (self.pose_frame_id != self.global_frame_id) and (not self.warned_frame_mismatch):
            self.warned_frame_mismatch = True
            self.get_logger().warn(
                f"pose frame={self.pose_frame_id} 与 global_path frame={self.global_frame_id} 不一致，"
                f"假定它们已经是同一 UTM 坐标系下的数值"
            )

        # 检查是否到达终点
        if self.reached_goal():
            self.publish_stop()
            return

        candidate = self.plan_once()
        if candidate is None:
            self.publish_stop()
            return

        # 发布局部路径 & 更新后的全局路径
        self.publish_local_path(candidate.poses)
        self.publish_updated_global_path(candidate)

        # 发布速度指令
        cmd = Twist()
        cmd.linear.x = candidate.linear_velocity
        cmd.angular.z = candidate.angular_velocity
        self.cmd_vel_pub.publish(cmd)

    # ================== 规划核心 ==================

    def plan_once(self) -> Optional[TrajectoryCandidate]:
        dynamic_window = self.compute_dynamic_window()
        if dynamic_window is None:
            return None

        target = self.find_target_point()
        if target is None:
            return None

        target_x, target_y, target_yaw, target_idx = target

        v_samples = self.sample_range(dynamic_window[0], dynamic_window[1],
                                      self.velocity_samples)
        w_samples = self.sample_range(dynamic_window[2], dynamic_window[3],
                                      self.yaw_samples)

        # lattice 偏置：保证有一组采样倾向于沿 global path 的曲率
        current_yaw = yaw_from_pose(self.current_pose)
        desired_yaw_rate = normalize_angle(target_yaw - current_yaw) / max(self.predict_time, 0.2)
        for bias in self.lattice_yaw_bias:
            w_des = desired_yaw_rate + bias
            w_des = max(-self.max_yaw_rate, min(self.max_yaw_rate, w_des))
            w_samples.append(w_des)

        best: Optional[TrajectoryCandidate] = None
        for v in v_samples:
            for w in w_samples:
                trajectory = self.simulate_trajectory(v, w)
                if not trajectory:
                    continue
                candidate = self.evaluate_candidate(trajectory, v, w, target)
                if candidate is None:
                    continue
                if best is None or candidate.score < best.score:
                    best = candidate
        return best

    def compute_dynamic_window(self) -> Optional[Tuple[float, float, float, float]]:
        if self.current_pose is None:
            return None

        # 当前速度（如果没有 odom，则默认 0）
        v = self.current_linear_vel if self.have_velocity else 0.0
        w = self.current_angular_vel if self.have_velocity else 0.0
        span = self.control_period

        v_min = max(self.min_speed, v - self.max_deceleration * span)
        v_max = min(self.max_speed, v + self.max_acceleration * span)
        w_min = max(-self.max_yaw_rate, w - self.max_yaw_acceleration * span)
        w_max = min(self.max_yaw_rate, w + self.max_yaw_acceleration * span)

        if v_min > v_max:
            v_min, v_max = v_max, v_min
        if w_min > w_max:
            w_min, w_max = w_max, w_min
        return (v_min, v_max, w_min, w_max)

    def sample_range(self, lower: float, upper: float, count: int) -> List[float]:
        if count <= 1:
            return [(lower + upper) * 0.5]
        if math.isclose(upper, lower):
            return [lower]
        step = (upper - lower) / float(count - 1)
        return [lower + i * step for i in range(count)]

    def simulate_trajectory(self, v: float, w: float) -> List[PoseStamped]:
        if self.current_pose is None:
            return []

        x = self.current_pose.position.x
        y = self.current_pose.position.y
        yaw = yaw_from_pose(self.current_pose)

        poses: List[PoseStamped] = []
        now = self.get_clock().now().to_msg()

        sim_steps = max(1, int(self.predict_time / self.predict_dt))
        for _ in range(sim_steps):
            x += v * math.cos(yaw) * self.predict_dt
            y += v * math.sin(yaw) * self.predict_dt
            yaw += w * self.predict_dt
            yaw = normalize_angle(yaw)

            pose = PoseStamped()
            pose.header.frame_id = self.global_frame_id
            pose.header.stamp = now
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            poses.append(pose)

        return poses

    def evaluate_candidate(self,
                           trajectory: List[PoseStamped],
                           v: float,
                           w: float,
                           target: Tuple[float, float, float, int]) -> Optional[TrajectoryCandidate]:
        if not trajectory:
            return None

        final_pose = trajectory[-1].pose
        final_x = final_pose.position.x
        final_y = final_pose.position.y
        final_yaw = yaw_from_pose(final_pose)

        target_x, target_y, target_yaw, target_idx = target

        # 航向误差
        heading_error = abs(normalize_angle(target_yaw - final_yaw)) / math.pi
        # 位置误差
        path_error = math.hypot(final_x - target_x, final_y - target_y)
        # 越快越好，因此速度“代价”选 (max_speed - |v|) / max_speed
        velocity_cost = (self.max_speed - abs(v)) / max(self.max_speed, 0.1)

        # 避障
        min_clearance, first_collision_time = self.estimate_clearance(trajectory)
        if min_clearance < self.robot_radius * 0.9:
            # 太近，直接丢弃轨迹
            return None

        clearance_cost = 0.0
        if min_clearance < self.robot_radius * 4.0:
            clearance_cost = 1.0 / max(min_clearance - self.robot_radius, 0.05)

        ttc_cost = 0.0
        if first_collision_time is not None:
            ttc_cost = 1.0 / max(first_collision_time, 0.05)

        score = (
            self.heading_weight * heading_error +
            self.path_follow_weight * (path_error / max(self.lookahead_distance, 0.1)) +
            self.velocity_weight * velocity_cost +
            self.clearance_weight * clearance_cost +
            self.ttc_weight * ttc_cost
        )

        return TrajectoryCandidate(
            linear_velocity=v,
            angular_velocity=w,
            poses=trajectory,
            min_clearance=min_clearance,
            first_collision_time=first_collision_time,
            heading_error=heading_error,
            path_error=path_error,
            score=score,
            target_index=target_idx,
        )

    def estimate_clearance(self, trajectory: List[PoseStamped]) -> Tuple[float, Optional[float]]:
        """
        沿轨迹扫描和障碍物的最小距离，以及第一次发生碰撞的时间（用于 TTC）。
        障碍物坐标在 UTM 系 self.global_obstacles 中。
        """
        if not self.global_obstacles:
            return float('inf'), None

        min_clearance = float('inf')
        first_collision_time: Optional[float] = None

        for idx, pose in enumerate(trajectory):
            px = pose.pose.position.x
            py = pose.pose.position.y
            for ox, oy in self.global_obstacles:
                dist = math.hypot(px - ox, py - oy)
                if dist < min_clearance:
                    min_clearance = dist
                if dist < self.robot_radius * 1.1 and first_collision_time is None:
                    first_collision_time = (idx + 1) * self.predict_dt
                    break
            if first_collision_time is not None:
                break

        return min_clearance, first_collision_time

    def find_target_point(self) -> Optional[Tuple[float, float, float, int]]:
        """
        在 global_path_points 中：
        1) 找离当前车最近的路径点（从 closest_path_idx 往后搜索）
        2) 从这个点开始沿路径前走 lookahead_distance，作为前视目标点
        """
        if not self.global_path_points or self.current_pose is None:
            return None

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        idx = self.closest_path_idx
        best_idx = idx
        best_dist = float('inf')

        # 先找最近点
        for i in range(idx, len(self.global_path_points)):
            px, py = self.global_path_points[i]
            dist = (px - robot_x) * (px - robot_x) + (py - robot_y) * (py - robot_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
            if dist <= 0.04:  # 距离很小就可以提前停止
                break
        self.closest_path_idx = best_idx

        # 再从最近点沿路径累计弧长，找前视点
        travel = 0.0
        last_x = robot_x
        last_y = robot_y
        target_idx = best_idx
        for i in range(best_idx, len(self.global_path_points)):
            px, py = self.global_path_points[i]
            travel += math.hypot(px - last_x, py - last_y)
            last_x = px
            last_y = py
            if travel >= self.lookahead_distance:
                target_idx = i
                break
        else:
            target_idx = len(self.global_path_points) - 1

        target_x, target_y = self.global_path_points[target_idx]
        # 目标朝向：当前车位到目标点的方向（也可以用路径切线，这里用简化版本）
        target_yaw = math.atan2(target_y - robot_y, target_x - robot_x)

        return target_x, target_y, target_yaw, target_idx

    def reached_goal(self) -> bool:
        if not self.global_path_points or self.current_pose is None:
            return False
        goal_x, goal_y = self.global_path_points[-1]
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        dist = math.hypot(goal_x - robot_x, goal_y - robot_y)
        return dist <= self.goal_tolerance

    # ================== 发布函数 ==================

    def publish_stop(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

        if self.current_pose is not None:
            pose_stamp = PoseStamped()
            pose_stamp.header.frame_id = self.global_frame_id
            pose_stamp.header.stamp = self.get_clock().now().to_msg()
            pose_stamp.pose = pose_copy(self.current_pose)

            self.publish_local_path([pose_stamp])

            dummy_candidate = TrajectoryCandidate(
                linear_velocity=0.0,
                angular_velocity=0.0,
                poses=[pose_stamp],
                min_clearance=float('inf'),
                first_collision_time=None,
                heading_error=0.0,
                path_error=0.0,
                score=0.0,
                target_index=self.closest_path_idx
            )
            self.publish_updated_global_path(dummy_candidate)

    def publish_local_path(self, poses: Sequence[PoseStamped]):
        path = Path()
        path.header.frame_id = self.global_frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = list(poses)
        self.local_path_pub.publish(path)

    def publish_updated_global_path(self, candidate: TrajectoryCandidate):
        """
        把当前选中的局部轨迹 + 全局路径剩余部分 拼成一个“更新后的全局路径”，方便可视化。
        """
        if not candidate.poses:
            return

        path = Path()
        path.header.frame_id = self.global_frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        combined: List[PoseStamped] = []
        # 先把局部轨迹复制进去
        for pose in candidate.poses:
            ps = PoseStamped()
            ps.header.frame_id = self.global_frame_id
            ps.header.stamp = path.header.stamp
            ps.pose = pose_copy(pose.pose)
            combined.append(ps)

        # 再拼接 global_path 剩余部分
        if self.global_path_msg is not None and candidate.target_index is not None:
            start_idx = min(max(candidate.target_index + 1, 0), len(self.global_path_msg.poses))
            for idx in range(start_idx, len(self.global_path_msg.poses)):
                pose_ext = PoseStamped()
                pose_ext.header.frame_id = self.global_frame_id
                pose_ext.header.stamp = path.header.stamp
                pose_ext.pose = pose_copy(self.global_path_msg.poses[idx].pose)
                combined.append(pose_ext)

        path.poses = combined
        
        # 输出提示
        print(f'publish updated global path: {len(path.poses)} points')
        
        self.updated_global_path_pub.publish(path)

    # ================== 坐标变换：自车系 -> UTM ==================

    def update_obstacle_world_coordinates(self):
        """
        将 self.relative_obstacles (自车系) 转换为 UTM 系 self.global_obstacles。
        假设 pose_utm 给出的 yaw 就是车辆在 UTM 下的朝向，
        且传感器坐标系与 base_link 差异很小，这里直接当成 base_link。
        """
        if not self.relative_obstacles or self.current_pose is None:
            self.global_obstacles = []
            return

        yaw = yaw_from_pose(self.current_pose)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        base_x = self.current_pose.position.x
        base_y = self.current_pose.position.y

        transformed: List[Tuple[float, float]] = []
        for rx, ry in self.relative_obstacles:
            gx = base_x + rx * cos_yaw - ry * sin_yaw
            gy = base_y + rx * sin_yaw + ry * cos_yaw
            transformed.append((gx, gy))

        self.global_obstacles = transformed


def main(args=None):
    rclpy.init(args=args)
    node = UtmLocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
