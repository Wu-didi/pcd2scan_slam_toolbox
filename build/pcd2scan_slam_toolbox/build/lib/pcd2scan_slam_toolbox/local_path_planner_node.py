#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid DWA/Lattice local planner.

订阅:
- /global_path (nav_msgs/Path)
- /pose (PoseStamped | PoseWithCovarianceStamped | Odometry)
- /odom (nav_msgs/Odometry)             # 可选，提供速度
- /scan (sensor_msgs/LaserScan)

发布:
- /local_path (nav_msgs/Path)           # 当前选中的局部轨迹
- /update_global_path (nav_msgs/Path)   # 避障后更新的全局路径
- /cmd_vel (geometry_msgs/Twist)        # 驱动控制指令

算法:
- 基于 Dynamic Window Approach 采样 (v, w)
- 将采样结果与一组 lattice 偏置结合，倾向于沿全局路径行驶
- 仿真若干秒并对候选轨迹打分：航向误差、路径偏差、速度、动态障碍清距、碰撞时间
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
)
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    while angle >= math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_pose(pose: Pose) -> float:
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


class DynamicLocalPlannerNode(Node):
    def __init__(self):
        super().__init__('local_path_planner_node')

        # --------- 常规参数 ---------
        self.declare_parameter('global_path_topic', 'global_path')
        self.declare_parameter('local_path_topic', 'local_path')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('update_global_path_topic', 'update_global_path')
        self.declare_parameter('pose_topic', 'amcl_pose')
        self.declare_parameter('pose_type', 'PoseWithCovarianceStamped')
        self.declare_parameter('velocity_topic', 'odom')
        self.declare_parameter('laser_topic', 'scan')
        self.declare_parameter('lookahead_distance', 6.0)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('robot_radius', 0.7)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('predict_time', 2.0)
        self.declare_parameter('predict_dt', 0.1)
        self.declare_parameter('velocity_samples', 8)
        self.declare_parameter('yaw_samples', 12)
        self.declare_parameter('max_speed', 2.0)
        self.declare_parameter('min_speed', 0.0)
        self.declare_parameter('max_yaw_rate', 1.2)
        self.declare_parameter('max_acceleration', 1.5)
        self.declare_parameter('max_deceleration', 2.0)
        self.declare_parameter('max_yaw_acceleration', 2.5)
        self.declare_parameter('path_follow_weight', 1.2)
        self.declare_parameter('heading_weight', 1.4)
        self.declare_parameter('velocity_weight', 0.4)
        self.declare_parameter('clearance_weight', 2.4)
        self.declare_parameter('ttc_weight', 1.0)
        self.declare_parameter('obstacle_max_range', 15.0)
        self.declare_parameter('dynamic_obstacle_horizon', 1.0)
        self.declare_parameter('lattice_yaw_bias', [-0.8, -0.4, 0.0, 0.4, 0.8])

        self.global_path_topic = self.get_parameter('global_path_topic').value
        self.local_path_topic = self.get_parameter('local_path_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.update_global_path_topic = self.get_parameter(
            'update_global_path_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.pose_type = str(self.get_parameter('pose_type').value).lower()
        self.velocity_topic = self.get_parameter('velocity_topic').value
        self.laser_topic = self.get_parameter('laser_topic').value
        self.lookahead_distance = float(
            self.get_parameter('lookahead_distance').value)
        self.goal_tolerance = float(
            self.get_parameter('goal_tolerance').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.control_frequency = max(
            1.0, float(self.get_parameter('control_frequency').value))
        self.predict_time = max(
            0.2, float(self.get_parameter('predict_time').value))
        self.predict_dt = max(
            0.02, float(self.get_parameter('predict_dt').value))
        self.velocity_samples = max(
            3, int(self.get_parameter('velocity_samples').value))
        self.yaw_samples = max(
            3, int(self.get_parameter('yaw_samples').value))
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.max_acceleration = float(
            self.get_parameter('max_acceleration').value)
        self.max_deceleration = float(
            self.get_parameter('max_deceleration').value)
        self.max_yaw_acceleration = float(
            self.get_parameter('max_yaw_acceleration').value)
        self.path_follow_weight = float(
            self.get_parameter('path_follow_weight').value)
        self.heading_weight = float(
            self.get_parameter('heading_weight').value)
        self.velocity_weight = float(
            self.get_parameter('velocity_weight').value)
        self.clearance_weight = float(
            self.get_parameter('clearance_weight').value)
        self.ttc_weight = float(self.get_parameter('ttc_weight').value)
        self.obstacle_max_range = float(
            self.get_parameter('obstacle_max_range').value)
        self.dynamic_obstacle_horizon = float(
            self.get_parameter('dynamic_obstacle_horizon').value)
        lattice_bias_param = self.get_parameter('lattice_yaw_bias').value
        self.lattice_yaw_bias = [
            float(val) for val in (lattice_bias_param or [])
        ]

        self.control_period = 1.0 / self.control_frequency
        self.steps_per_prediction = max(1,
                                        int(self.predict_time / self.predict_dt))

        # --------- 状态 ---------
        self.global_path_msg: Optional[Path] = None
        self.global_path_points: List[Tuple[float, float]] = []
        self.global_frame_id: str = 'map'
        self.closest_path_idx = 0

        self.current_pose: Optional[Pose] = None
        self.pose_frame_id: str = 'map'
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        self.relative_obstacles: List[Tuple[float, float]] = []
        self.global_obstacles: List[Tuple[float, float]] = []
        self.last_scan_ranges: Optional[List[float]] = None
        self.last_scan_time = None

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

        # --------- 订阅定位 ---------
        self.pose_sub = self.create_pose_subscription()
        self.odom_sub = None
        if 'odom' not in self.pose_type:
            self.odom_sub = self.create_subscription(
                Odometry,
                self.velocity_topic,
                self.velocity_callback,
                10
            )

        # --------- 激光雷达 ---------
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.laser_topic,
            self.scan_callback,
            10
        )

        # --------- 发布 ---------
        self.local_path_pub = self.create_publisher(Path,
                                                    self.local_path_topic,
                                                    10)
        self.cmd_vel_pub = self.create_publisher(Twist,
                                                 self.cmd_vel_topic,
                                                 10)
        self.updated_global_path_pub = self.create_publisher(
            Path,
            self.update_global_path_topic,
            10
        )

        # --------- 控制循环 ---------
        self.timer = self.create_timer(self.control_period,
                                       self.control_loop)

        self.get_logger().info(
            "Hybrid 区域规划器启动: 订阅 global_path=%s pose=%s "
            "odom=%s scan=%s"
            % (self.global_path_topic,
               self.pose_topic,
               self.velocity_topic,
               self.laser_topic)
        )

    # ================== 订阅回调 ==================

    def global_path_callback(self, msg: Path):
        self.global_path_msg = msg
        self.global_frame_id = msg.header.frame_id or 'map'
        self.global_path_points = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
        self.closest_path_idx = 0
        self.get_logger().info(
            f"收到 global_path: {len(self.global_path_points)} points, frame={self.global_frame_id}")

    def pose_stamped_callback(self, msg: PoseStamped):
        self.current_pose = pose_copy(msg.pose)
        self.pose_frame_id = msg.header.frame_id or 'map'
        self.update_obstacle_world_coordinates()

    def pose_with_cov_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = pose_copy(msg.pose.pose)
        self.pose_frame_id = msg.header.frame_id or 'map'
        self.update_obstacle_world_coordinates()

    def odom_pose_callback(self, msg: Odometry):
        self.current_pose = pose_copy(msg.pose.pose)
        self.pose_frame_id = msg.header.frame_id or 'odom'
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z
        self.update_obstacle_world_coordinates()

    def velocity_callback(self, msg: Odometry):
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z

    def scan_callback(self, msg: LaserScan):
        ranges = msg.ranges
        now = self.get_clock().now()
        dt = 0.0
        if self.last_scan_time is not None:
            dt = (now - self.last_scan_time).nanoseconds * 1e-9
        self.last_scan_time = now

        predicted: List[Tuple[float, float]] = []
        n = len(ranges)
        for i, rng in enumerate(ranges):
            if math.isinf(rng) or math.isnan(rng):
                continue
            if rng > self.obstacle_max_range:
                continue
            angle = msg.angle_min + i * msg.angle_increment

            projected = rng
            if dt > 1e-3 and self.last_scan_ranges is not None and i < len(self.last_scan_ranges):
                prev = self.last_scan_ranges[i]
                if prev is not None and not math.isinf(prev) and not math.isnan(prev):
                    range_rate = (rng - prev) / dt
                    projected = rng + \
                        range_rate * self.dynamic_obstacle_horizon
                    projected = max(projected, 0.05)

            px = projected * math.cos(angle)
            py = projected * math.sin(angle)
            predicted.append((px, py))

        self.relative_obstacles = predicted
        self.last_scan_ranges = list(ranges)
        self.update_obstacle_world_coordinates()

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
                self.odom_pose_callback,
                10
            )
        return self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_stamped_callback,
            10
        )

    # ================== 核心控制循环 ==================

    def control_loop(self):
        if not self.global_path_points or self.current_pose is None:
            return
        if self.pose_frame_id != self.global_frame_id and not self.warned_frame_mismatch:
            self.warned_frame_mismatch = True
            self.get_logger().warn(
                f"定位 frame={self.pose_frame_id} 与全局 frame={self.global_frame_id} 不一致，假定已经在同一坐标系")

        if self.reached_goal():
            self.publish_stop()
            return

        candidate = self.plan_once()
        if candidate is None:
            self.publish_stop()
            return

        self.publish_local_path(candidate.poses)
        self.publish_updated_global_path(candidate)

        cmd = Twist()
        cmd.linear.x = candidate.linear_velocity
        cmd.angular.z = candidate.angular_velocity
        self.cmd_vel_pub.publish(cmd)

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

        # lattice 偏置，确保接近全局曲率
        desired_yaw_rate = normalize_angle(target_yaw - yaw_from_pose(
            self.current_pose)) / max(self.predict_time, 0.2)
        for bias in self.lattice_yaw_bias:
            w_samples.append(
                max(-self.max_yaw_rate,
                    min(self.max_yaw_rate, desired_yaw_rate + bias)))

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

        v = self.current_linear_vel
        w = self.current_angular_vel
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

        sim_steps = int(self.predict_time / self.predict_dt)
        for step_idx in range(sim_steps):
            x += v * math.cos(yaw) * self.predict_dt
            y += v * math.sin(yaw) * self.predict_dt
            yaw += w * self.predict_dt
            yaw = normalize_angle(yaw)

            pose = PoseStamped()
            pose.header.frame_id = self.global_frame_id
            pose.header.stamp = now
            pose.pose.position.x = x
            pose.pose.position.y = y
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

        heading_error = abs(normalize_angle(target_yaw -
                                            final_yaw)) / math.pi
        path_error = math.hypot(final_x - target_x, final_y - target_y)
        velocity_cost = (self.max_speed - abs(v)) / max(self.max_speed, 0.1)

        min_clearance, first_collision_time = self.estimate_clearance(trajectory)
        if min_clearance < self.robot_radius * 0.9:
            return None

        clearance_cost = 0.0
        if min_clearance < self.robot_radius * 4.0:
            clearance_cost = 1.0 / max(min_clearance - self.robot_radius, 0.05)

        ttc_cost = 0.0
        if first_collision_time is not None:
            ttc_cost = 1.0 / max(first_collision_time, 0.05)

        score = (
            self.heading_weight * heading_error +
            self.path_follow_weight *
            (path_error / max(self.lookahead_distance, 0.1)) +
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
        if not self.global_path_points:
            return None
        if self.current_pose is None:
            return None

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        idx = self.closest_path_idx
        best_idx = idx
        best_dist = float('inf')
        for i in range(idx, len(self.global_path_points)):
            px, py = self.global_path_points[i]
            dist = (px - robot_x) * (px - robot_x) + \
                (py - robot_y) * (py - robot_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
            if dist <= 0.04:
                break
        self.closest_path_idx = best_idx

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
        target_yaw = math.atan2(
            target_y - robot_y,
            target_x - robot_x
        )
        return target_x, target_y, target_yaw, target_idx

    def reached_goal(self) -> bool:
        if not self.global_path_points or self.current_pose is None:
            return False
        goal_x, goal_y = self.global_path_points[-1]
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        dist = math.hypot(goal_x - robot_x, goal_y - robot_y)
        return dist <= self.goal_tolerance

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
        if not candidate.poses:
            return

        path = Path()
        path.header.frame_id = self.global_frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        combined: List[PoseStamped] = []
        for pose in candidate.poses:
            pose_copy_stamp = PoseStamped()
            pose_copy_stamp.header.frame_id = self.global_frame_id
            pose_copy_stamp.header.stamp = path.header.stamp
            pose_copy_stamp.pose = pose_copy(pose.pose)
            combined.append(pose_copy_stamp)

        if self.global_path_msg is not None and candidate.target_index is not None:
            start_idx = min(
                max(candidate.target_index + 1, 0), len(self.global_path_msg.poses))
            for idx in range(start_idx, len(self.global_path_msg.poses)):
                pose_ext = PoseStamped()
                pose_ext.header.frame_id = self.global_frame_id
                pose_ext.header.stamp = path.header.stamp
                pose_ext.pose = pose_copy(self.global_path_msg.poses[idx].pose)
                combined.append(pose_ext)

        path.poses = combined
        self.updated_global_path_pub.publish(path)

    def update_obstacle_world_coordinates(self):
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
    node = DynamicLocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
