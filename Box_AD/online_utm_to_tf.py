#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu, NavSatFix
import tf2_ros
from pyproj import CRS, Transformer

# 经纬度 -> UTM 的投影
wgs84 = CRS("EPSG:4326")
utm_zone_number = 49  # 换成你实际的区
utm_crs = CRS(f"EPSG:{32600 + utm_zone_number}")
projector_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)


def latlon_to_utm(lon, lat):
    x, y = projector_to_utm.transform(lon, lat)
    return x, y


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


class UTMToTF(Node):
    def __init__(self):
        super().__init__('utm_to_tf')

        # 订阅 IMU 和 GPS
        self.create_subscription(Imu, 'IMU', self.imu_callback, 10)
        self.create_subscription(NavSatFix, 'gps/fix', self.gps_callback, 10)

        # TF 广播器
        self.br = tf2_ros.TransformBroadcaster(self)

        # 偏移量
        self.OFFSET_X = 1240249.0084191752
        self.OFFSET_Y = 3555312.0655697277

        # 坐标系
        self.parent_frame = 'odom'
        self.child_frame = 'base_link'

        # 保存最新的 yaw
        self.yaw = 0.0

    def imu_callback(self, msg: Imu):
        # 四元数 -> 欧拉 -> yaw
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = self.quaternion_to_euler(x, y, z, w)
        self.yaw = yaw

    def gps_callback(self, msg: NavSatFix):
        # 经纬度 -> UTM
        utm_x, utm_y = latlon_to_utm(msg.longitude, msg.latitude)
        x = utm_x - self.OFFSET_X
        y = utm_y - self.OFFSET_Y
        z = msg.altitude if not math.isnan(msg.altitude) else 0.0

        # 这里改成用“现在时间”
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()   # ★ 改这里
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, self.yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.br.sendTransform(t)

        self.get_logger().info(
            f'pub TF at NOW x={x:.3f} y={y:.3f} yaw={math.degrees(self.yaw):.1f}'
        )

    @staticmethod
    def quaternion_to_euler(x, y, z, w):
        roll = math.atan2(2.0 * (w * x + y * z),
                          1.0 - 2.0 * (x * x + y * y))
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        yaw = math.atan2(2.0 * (w * z + x * y),
                         1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw


def main():
    rclpy.init()
    node = UTMToTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
