#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

import sys
# 添加模块所在路径
sys.path.append('/home/nvidia/vcii/wudi/pcd2scan_slam_toolbox/pcd2scan_slam_toolbox')

from can_use import Can_use

class UTMToTF(Node):
    def __init__(self):
        super().__init__('utm_to_tf')

        # 初始can-use, 用于获取 UTM 位姿
        self.can_use = Can_use()
        
        # 创建 TF 广播器
        self.br = tf2_ros.TransformBroadcaster(self)

        # 假定的 UTM 位姿，后面你自己改
        self.x = 0.0
        self.y = 0.0
        self.yaw_deg = 0.0
        
        self.OFFSET_X = 1240249.0084191752
        self.OFFSET_Y = 3555312.0655697277

        # 坐标系名字
        self.parent_frame = 'odom'
        self.child_frame = 'base_link'

        # 10Hz 定时发
        self.timer = self.create_timer(0.1, self.timer_cb)

    def timer_cb(self):
        yaw = math.radians(self.yaw_deg)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        self.can_use.read_ins_info()
        while not self.can_use.flag or self.can_use.ego_x == 0:
            self.can_use.read_ins_info()
            # print("searching coord....")
        print("utm coord:",self.can_use.ego_x,self.can_use.ego_y)
        self.x = self.can_use.ego_x-self.OFFSET_X
        self.y = self.can_use.ego_y-self.OFFSET_Y
        print("ego coordinate:",self.x,self.y)
        yaw = self.can_use.ego_yaw_rad


        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)

        self.br.sendTransform(t)

def main():
    rclpy.init()
    print("=== utm_to_tf node start ===")
    node = UTMToTF()
    rclpy.spin(node)          # 关键：让节点一直跑
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
