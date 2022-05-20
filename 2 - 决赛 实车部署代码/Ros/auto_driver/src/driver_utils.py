#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import rospy
import numpy as np
import math
from std_msgs.msg import Int32, Float32
from bluetooth_bridge.msg import Sensors, HilensMsg, LaneMsg, BoardMsg
import threading

def loop_idx(size):
    """ 生成循环索引 """
    idx = 0
    while True:
        yield idx
        idx = 0 if idx>=size-1 else idx+1

class Driver(object):
    """
    小车驱动类,包含以下功能：
    驱动：
    1. set_direction: [0,100]
    2. set_speed: [0,100]
    3. set_mode: [1D, 2N, 3P, 4R]
    4. set_beep: [0,1]
    读取传感器：
    1. get_acc: imu加速度
    2. get_alpha: imu角加速度
    3. get_B:  imu磁场
    4. get_theta: imu角度
    5. get_speed: 实际电机速度
    6. get_mode: 实际档位
    7. get_direction: 实际方向
    8. get_supersonic: 实际超声波距离
    9. get_sensors: 所有传感器信息
    10. get_lane: 小车偏离路中心距离
    11. get_objs: 目标检测结果
    """
    def __init__(self, cache_size=10, debug=True):
        self.Debug = debug
        # 控制
        self.direction_pub  = rospy.Publisher("/auto_driver/send/direction", Int32, queue_size=10)
        self.speed_pub      = rospy.Publisher("/auto_driver/send/speed", Int32, queue_size=10)
        self.mode_pub       = rospy.Publisher("/auto_driver/send/mode", Int32, queue_size=10)
        self.beep_pub      = rospy.Publisher("/auto_driver/send/beep", Int32, queue_size=10)
        # 传感
        self.sensor_sub    = rospy.Subscriber('/vcu', Sensors, self.sensors_callback)
        self.hilens_sub    = rospy.Subscriber('/hilens', HilensMsg, self.hilens_callback)
        self.lane_sub      = rospy.Subscriber('/lane_detect', LaneMsg, self.lane_callback)
        self.lidar_sub     = rospy.Subscriber('/lidar', BoardMsg, self.lidar_callback)

        self.cache_size = cache_size
        self.sensor_cache = [Sensors(), ] * self.cache_size
        self.hilens_cache = [HilensMsg(), ] * self.cache_size
        self.lane_cache = [LaneMsg(), ] * self.cache_size
        self.lidar_cache = [BoardMsg(), ]* self.cache_size
        self.loop_idx = loop_idx(self.cache_size)
        self.hi_loop_idx = loop_idx(self.cache_size)
        self.lane_loop_idx = loop_idx(self.cache_size)
        self.lidar_loop_idx = loop_idx(self.cache_size)
        self.idx = 0
        self.hi_idx = 0
        self.lane_idx = 0
        self.lidar_idx = 0
        self.ros_spin = threading.Thread(target = rospy.spin)
        self.ros_spin.setDaemon(True)
        self.ros_spin.start()
        time.sleep(1.5)
        rospy.loginfo("[Driver]: Init" )


    #--- 设置
    def set_direction(self, x):
        assert x>=0 and x<=100, 'direction must in [0, 100]'
        if not rospy.is_shutdown():
            self.direction_pub.publish(x)
            if self.Debug:
                rospy.loginfo("[Driver]: set direction to %d" % x )

    def set_speed(self, x):
        assert x>=0 and x<=100, 'speed must in [0, 100]'
        if not rospy.is_shutdown():
            self.speed_pub.publish(x)
            if self.Debug:
                rospy.loginfo("[Driver]: set speed to %d" % x )

    def set_mode(self, x):
        """ [1D, 2N, 3P, 4R] """
        if isinstance(x, str):
            mode_dict = {'D':1, 'N':2, 'P':3, 'R':4}
            x = mode_dict.get(x, 0)
        assert x>=1 and x<=4, 'mode must in [1, 4]'
        if not rospy.is_shutdown():
            self.mode_pub.publish(x)
            if self.Debug:
                rospy.loginfo("[Driver]: set mode to %d" % x )

    def set_beep(self, x):
        assert x>=0 and x<=1, 'beep must in [0, 1]'
        if not rospy.is_shutdown():
            self.beep_pub.publish(x)
            if self.Debug:
                rospy.loginfo("[Driver]: set beep to %d" % x )

    #--- 回调函数
    def sensors_callback(self, data):
        tmp = self.loop_idx.next()
        self.sensor_cache[tmp] = data
        self.idx = tmp

    def hilens_callback(self, data):
        tmp = self.hi_loop_idx.next()
        self.hilens_cache[tmp] = data
        self.hi_idx = tmp

    def lane_callback(self, data):
        tmp = self.lane_loop_idx.next()
        self.lane_cache[tmp] = data
        self.lane_idx = tmp

    def lidar_callback(self, data):
        tmp = self.lidar_loop_idx.next()
        self.lidar_cache[tmp] = data
        self.lidar_idx = tmp

    #--- 获取小车状态
    def get_lane(self):
        return  self.lane_cache[self.lane_idx].bias, \
                self.lane_cache[self.lane_idx].gear

    def get_objs(self, index):  # index: ["green_go", "pedestrian_crossing", "red_stop", "speed_limited", "speed_minimum", "speed_unlimited", "yellow_back"]
        flag = self.hilens_cache[self.hi_idx].flag[index]
        x_min = self.hilens_cache[self.hi_idx].x_min[index]
        x_max = self.hilens_cache[self.hi_idx].x_max[index]
        y_min = self.hilens_cache[self.hi_idx].y_min[index]
        y_max = self.hilens_cache[self.hi_idx].y_max[index]
        score = self.hilens_cache[self.hi_idx].score[index]
        return flag, x_min, x_max, y_min, y_max, score

    def get_lidar(self):
        return self.lidar_cache[self.lidar_idx].is_obstacle, \
               self.lidar_cache[self.lidar_idx].is_board, \
               self.lidar_cache[self.lidar_idx].pos_bias, \
               self.lidar_cache[self.lidar_idx].angle_bias

    def get_acc(self):
        return self.sensor_cache[self.idx].ax, \
               self.sensor_cache[self.idx].ay, \
               self.sensor_cache[self.idx].az

    def get_alpha(self):
        return  self.sensor_cache[self.idx].alphax, \
                self.sensor_cache[self.idx].alphay, \
                self.sensor_cache[self.idx].alphaz

    def get_B(self):
        return  self.sensor_cache[self.idx].Bx, \
                self.sensor_cache[self.idx].By, \
                self.sensor_cache[self.idx].Bz

    def get_theta(self):
        return  self.sensor_cache[self.idx].thetax, \
                self.sensor_cache[self.idx].thetay, \
                self.sensor_cache[self.idx].thetaz

    def get_speed(self):
        return  self.sensor_cache[self.idx].MotorSpeed

    def get_mode(self):
        return  self.sensor_cache[self.idx].Mode

    def get_direction(self):
        return  self.sensor_cache[self.idx].Direction

    def get_supersonic(self):
        return  self.sensor_cache[self.idx].Supersonic

    def get_sensors(self):
        return  self.sensor_cache[self.idx]

