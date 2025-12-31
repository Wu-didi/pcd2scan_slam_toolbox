#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UTM-based local planner with simple lattice paths (no controller).

订阅:
- /global_path_utm (nav_msgs/Path, frame_id="utm")
- /pose_utm       (geometry_msgs/PoseStamped, frame_id="utm")
- /fast_obstacles (std_msgs/String, JSON, 障碍物在自车坐标系下)
- /odom           (nav_msgs/Odometry，可选，仅用于获取当前速度；本实现未使用，可留作扩展)

发布:
- /local_path         (nav_msgs/Path)          # 当前选中的局部 lattice 轨迹
- /update_global_path (nav_msgs/Path)          # 局部轨迹 + 剩余全局路径
- /cmd_vel            (geometry_msgs/Twist)    # 这里只在“停车”时发 0，正常运行你可以不订阅

算法（简化 lattice）:
- 在全局路径附近，构造多条“平行 offset 轨迹”：
  * lateral_offsets = [-2, -1, 0, 1, 2] 米（参数可调）
  * 每条轨迹 = 将全局路径某段按法向量平移 offset
- 对每条轨迹做障碍物碰撞检测（fast_obstacles 变换到 UTM）
- 用简单代价：横向偏移 + 障碍物间距 + 与全局终点的距离，选出最优轨迹
- 不做任何速度/转角控制，仅发布路径，供下游 MPC / Pure Pursuit 使用
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


# ================== 工具函数 ==================

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
class LatticeCandidate:
    offset: float                     # 横向偏移（米）
    poses: List[PoseStamped]          # 该候选局部路径
    score: float                      # 代价
    end_index: int                    # 在 global_path 中对应的最后一个索引


# ================== 主节点 ==================

class UtmLatticeLocalPlannerNode(Node):
    def __init__(self):
        super().__init__('lattice_local_planner_node')

        # --------- 话题参数 ---------
        self.declare_parameter('global_path_topic', 'global_path_utm')
        self.declare_parameter('pose_topic', 'pose_utm')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('local_path_topic', 'local_path')
        self.declare_parameter('update_global_path_topic', 'update_global_path')
        self.declare_parameter('obstacle_topic', 'fast_obstacles')
        self.declare_parameter('velocity_topic', 'odom')
        self.declare_parameter('use_odom_velocity', True)

        # --------- 路径参数 ---------
        self.declare_parameter('lookahead_distance', 10.0)   # 用于最近点搜索的舒适范围
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('robot_radius', 0.7)
        self.declare_parameter('control_frequency', 10.0)

        # --------- lattice 采样参数 ---------
        self.declare_parameter('forward_length', 30.0)        # 向前生成多长的局部路径（米）
        self.declare_parameter('lateral_offsets', [-2.0, -1.0, 0.0, 1.0, 2.0])

        # --------- 代价权重 ---------
        self.declare_parameter('offset_weight', 1.0)          # 偏移越小越好
        self.declare_parameter('clearance_weight', 1.0)       # 离障碍越远越好
        self.declare_parameter('goal_weight', 0.2)            # 终点越接近全局终点越好

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

        self.forward_length = float(self.get_parameter('forward_length').value)
        lateral_param = self.get_parameter('lateral_offsets').value
        self.lateral_offsets: List[float] = [float(v) for v in (lateral_param or [])]

        self.offset_weight = float(self.get_parameter('offset_weight').value)
        self.clearance_weight = float(self.get_parameter('clearance_weight').value)
        self.goal_weight = float(self.get_parameter('goal_weight').value)

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

        # --------- 订阅 ---------
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

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_callback,
            10
        )

        if self.use_odom_velocity:
            self.odom_sub = self.create_subscription(
                Odometry,
                self.velocity_topic,
                self.odom_callback,
                10
            )
        else:
            self.odom_sub = None

        self.obstacle_sub = self.create_subscription(
            String,
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
            "Simple lattice UTM Local Planner (no controller) 启动: "
            f"global_path_topic={self.global_path_topic}, "
            f"pose_topic={self.pose_topic}, "
            f"obstacle_topic={self.obstacle_topic}, "
            f"velocity_topic={self.velocity_topic}, "
            f"use_odom_velocity={self.use_odom_velocity}"
        )

    # ================== 回调 ==================

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

        if (self.pose_frame_id != self.global_frame_id) and (not self.warned_frame_mismatch):
            self.warned_frame_mismatch = True
            self.get_logger().warn(
                f"pose frame={self.pose_frame_id} 与 global_path frame={self.global_frame_id} 不一致，"
                f"假定它们已经是同一 UTM 坐标系下的数值"
            )

    def odom_callback(self, msg: Odometry):
        # 当前版本的 lattice 不用速度，但保留接口
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z
        self.have_velocity = True

    def fast_obstacle_callback(self, msg: String):
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

        self.update_obstacle_world_coordinates()

    # ================== 控制循环（只做路径规划） ==================

    def control_loop(self):
        if not self.global_path_points or self.current_pose is None:
            return

        if self.reached_goal():
            self.publish_stop()
            return

        # 1) 找最近的全局路径索引
        start_idx = self.find_closest_path_index()

        # 2) 构造多条 lattice 候选路径
        candidates = self.build_lattice_candidates(start_idx)
        if not candidates:
            self.get_logger().warn("没有可用的 lattice 轨迹候选，停车")
            self.publish_stop()
            return

        # 3) 选最优
        best = min(candidates, key=lambda c: c.score)

        # 4) 发布局部路径与更新后的全局路径
        self.publish_local_path(best.poses)
        self.publish_updated_global_path(best)
        # 不再在这里发布速度，由你的 MPC/PP 使用 local_path 控制车辆

    # ================== lattice 构造与评估 ==================

    def find_closest_path_index(self) -> int:
        if not self.global_path_points or self.current_pose is None:
            return 0

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        idx0 = self.closest_path_idx
        best_idx = idx0
        best_dist = float('inf')

        for i in range(idx0, len(self.global_path_points)):
            px, py = self.global_path_points[i]
            dist = (px - robot_x) ** 2 + (py - robot_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
            if dist <= 0.04:
                break

        self.closest_path_idx = best_idx
        return best_idx

    def build_lattice_candidates(self, start_idx: int) -> List[LatticeCandidate]:
        if not self.global_path_points:
            return []

        candidates: List[LatticeCandidate] = []

        for offset in self.lateral_offsets:
            poses, end_idx, min_clearance, collision = self.build_single_lattice_path(
                start_idx, offset
            )
            if not poses:
                continue
            if collision:
                # 发生碰撞则丢弃
                continue

            # 代价：偏移 + 障碍物 + 与全局终点距离
            path_score = self.compute_lattice_score(
                offset, poses, end_idx, min_clearance
            )

            candidates.append(
                LatticeCandidate(
                    offset=offset,
                    poses=poses,
                    score=path_score,
                    end_index=end_idx
                )
            )

        return candidates

    def build_single_lattice_path(
        self, start_idx: int, offset: float
    ) -> Tuple[List[PoseStamped], int, float, bool]:
        """
        沿 global_path，从 start_idx 开始，向前累积 arc-length，直到 forward_length，
        对每个路径点按照路径切线方向的法向量平移 offset，得到一条平行路径。
        返回：
        - 该路径的 Pose 列表
        - 对应的末尾索引 end_idx
        - 该路径上距离障碍物的最小间距
        - 是否与障碍物发生碰撞（<= robot_radius）
        """
        if start_idx >= len(self.global_path_points) - 1:
            return [], start_idx, float('inf'), False

        path_points: List[Tuple[float, float]] = []
        travel = 0.0
        last_x, last_y = self.global_path_points[start_idx]

        end_idx = start_idx

        for i in range(start_idx, len(self.global_path_points) - 1):
            x, y = self.global_path_points[i]
            x_next, y_next = self.global_path_points[i + 1]

            if i == start_idx:
                travel = 0.0
            else:
                travel += math.hypot(x - last_x, y - last_y)
            last_x, last_y = x, y

            if travel > self.forward_length:
                end_idx = i
                break

            # 路径切线方向
            theta = math.atan2(y_next - y, x_next - x)
            # 法向量 (左侧) = [-sin(theta), cos(theta)]
            nx = -math.sin(theta)
            ny = math.cos(theta)

            # 平移 offset
            x_off = x + offset * nx
            y_off = y + offset * ny

            path_points.append((x_off, y_off))
            end_idx = i

        if not path_points:
            return [], start_idx, float('inf'), False

        poses: List[PoseStamped] = []
        stamp = self.get_clock().now().to_msg()
        min_clearance = float('inf')
        collision = False

        for (px, py) in path_points:
            pose = PoseStamped()
            pose.header.frame_id = self.global_frame_id
            pose.header.stamp = stamp
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            poses.append(pose)

            # 和障碍物的距离
            if self.global_obstacles:
                for ox, oy in self.global_obstacles:
                    dist = math.hypot(px - ox, py - oy)
                    if dist < min_clearance:
                        min_clearance = dist
                    if dist <= self.robot_radius:
                        collision = True
                        break
                if collision:
                    break

        if not self.global_obstacles:
            min_clearance = float('inf')

        return poses, end_idx, min_clearance, collision

    def compute_lattice_score(
        self,
        offset: float,
        poses: List[PoseStamped],
        end_idx: int,
        min_clearance: float
    ) -> float:
        """
        简单代价：
        - 偏移越小越好（绝对值）
        - 离障碍越远越好（1/min_clearance）
        - 局部路径终点越接近全局终点越好
        """
        offset_cost = abs(offset)

        if min_clearance == float('inf'):
            clearance_cost = 0.0
        else:
            clearance_cost = 1.0 / max(min_clearance, 0.01)

        goal_cost = 0.0
        if self.global_path_points and poses:
            goal_x, goal_y = self.global_path_points[-1]
            end_pose = poses[-1].pose
            ex = end_pose.position.x
            ey = end_pose.position.y
            goal_cost = math.hypot(goal_x - ex, goal_y - ey)

        score = (
            self.offset_weight * offset_cost +
            self.clearance_weight * clearance_cost +
            self.goal_weight * goal_cost
        )
        return score

    # ================== 终点判断 ==================

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
        # 停车：发 0 cmd_vel（下游不用订阅的话也无所谓）
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

        if self.current_pose is not None:
            pose_stamp = PoseStamped()
            pose_stamp.header.frame_id = self.global_frame_id
            pose_stamp.header.stamp = self.get_clock().now().to_msg()
            pose_stamp.pose = pose_copy(self.current_pose)

            self.publish_local_path([pose_stamp])

            dummy_candidate = LatticeCandidate(
                offset=0.0,
                poses=[pose_stamp],
                score=0.0,
                end_index=self.closest_path_idx
            )
            self.publish_updated_global_path(dummy_candidate)

    def publish_local_path(self, poses: Sequence[PoseStamped]):
        path = Path()
        path.header.frame_id = self.global_frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = list(poses)
        self.local_path_pub.publish(path)

    def publish_updated_global_path(self, candidate: LatticeCandidate):
        if not candidate.poses:
            return

        path = Path()
        path.header.frame_id = self.global_frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        combined: List[PoseStamped] = []

        # 局部路径
        for pose in candidate.poses:
            ps = PoseStamped()
            ps.header.frame_id = self.global_frame_id
            ps.header.stamp = path.header.stamp
            ps.pose = pose_copy(pose.pose)
            combined.append(ps)

        # 全局剩余路径
        if self.global_path_msg is not None:
            start_idx = min(
                max(candidate.end_index + 1, 0),
                len(self.global_path_msg.poses)
            )
            for idx in range(start_idx, len(self.global_path_msg.poses)):
                pose_ext = PoseStamped()
                pose_ext.header.frame_id = self.global_frame_id
                pose_ext.header.stamp = path.header.stamp
                pose_ext.pose = pose_copy(self.global_path_msg.poses[idx].pose)
                combined.append(pose_ext)

        path.poses = combined

        print(f'publish updated global path (lattice): {len(path.poses)} points')
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
    node = UtmLatticeLocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
