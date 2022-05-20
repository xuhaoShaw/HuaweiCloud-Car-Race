#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import time
import numpy as np
# from std_msgs.msg import Int32, String
from sensor_msgs.msg import LaserScan
from bluetooth_bridge.msg import BoardMsg
import threading
import cv2

def rad2deg(rad):
    return rad / np.pi * 180

class lidar_node():
    """ 激光雷达检测障碍物和挡板类
    实现功能：
    1. 接收 /scan 数据绘制图像
    2. 检测以小车为中心一个长方形范围内的障碍物
    3. 检测两侧挡板
    """
    def __init__(self, ):
        rospy.init_node('lidar_node', anonymous=False)
        self.lidar_pub = rospy.Publisher('/lidar', BoardMsg, queue_size=100)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.callback_scan)
        self.imSize = (520, 520)
        self.lidar_img = np.zeros(self.imSize) # 雷达图像
        self.line1 = np.array([0, 0])  # 左挡板和右挡板
        self.line2 = np.array([0, 0])
        self.line_mid = np.array([0, 0])
        self.pos_bias = 0   # 位置偏差，>0表示小车偏右
        self.ang_bias = 0   # 角度偏差，>0表示小车偏右
        self.center = (260, 260)
        self.roi = (80, 80, 20)  # halfwidth, forward, backward
        # 障碍物检测参数
        self.obs_distance = 0.8
        self.obs_range = (160, 200)
        self.obs_threshold = 10
        self.obs_check_width = 20
        self.obs_check_range = (self.obs_range[0] - self.obs_check_width, self.obs_range[1] + self.obs_check_width)
        self.obs_exist = 0

        self.halfwidth, self.forward, self.backward = self.roi
        self.board_exist = 0
        self.ros_spin = threading.Thread(target = rospy.spin)
        self.ros_spin.setDaemon(True)
        self.ros_spin.start()
        time.sleep(1)
        print("[lidar_board]: Init")

    def callback_scan(self, scan):
        """ 将/scan接收到的数据装换成图片保存 """
        img = np.zeros(self.imSize, dtype=np.uint8)
        ranges = np.array(scan.ranges)  # [1440,]
        ranges[ranges>2.5] = 0
        angle = np.arange(len(scan.ranges)) * scan.angle_increment
        px = ( np.cos(angle) * ranges * 100 ).astype(np.int32) + 260
        py = ( np.sin(angle) * ranges * 100 ).astype(np.int32) + 260
        img[px, py] = 255
        self.lidar_img = img
        self.board_det() # 更新 board_exist, pos_bias, ang_bias
        self.obj_det(scan) # 更新障碍物obj_exist
        self.lidar_pub.publish(BoardMsg(    is_obstacle = self.obs_exist,
                                            flag = self.board_exist,
                                            bias = self.pos_bias,
                                            angle = rad2deg(self.ang_bias)))


    def obj_det(self, scan):
        angle_increment_ = scan.angle_increment * 180 / np.pi

        ranges = np.array(scan.ranges)  # [1440,]
        ranges_obs = ranges
        ranges_obs[ranges_obs > 2.5] = 0

        angle_1 = self.obs_range[0]
        angle_2 = self.obs_range[1]
        angle_1_num = int(angle_1 / angle_increment_)
        angle_2_num = int(angle_2 / angle_increment_)
        #angle_1to2 = np.arange(angle_1_num, angle_2_num) * scan.angle_increment
        ranges_1to2 = ranges_obs[angle_1_num:angle_2_num]
        check_ranges = ranges_obs[int(self.obs_check_range[0] / angle_increment_):int(self.obs_check_range[1] / angle_increment_)]
        # print(angle_increment_)
        # print(angle_1to2)
        # px = (np.cos(angle_1to2) * ranges_1to2 * 100).astype(np.int32) + 260
        # py = (np.sin(angle_1to2) * ranges_1to2 * 100).astype(np.int32) + 260
        # img2 = np.zeros(imSize, dtype=np.uint8)
        # img2[px, py] = 255

        ranges_1to2[ranges_1to2 > self.obs_distance] = 0
        # 设置障碍物检测范围检查机制
        check_ranges[check_ranges > self.obs_distance] = 0
        y = np.nonzero(ranges_1to2)
        if len(y[0]) > self.obs_threshold:
            self.obs_exist = 1
        else:
            self.obs_exist = 0


    def board_det(self):
        # 剪裁
        xc, yc = self.center
        img_roi = self.lidar_img[yc-self.roi[1]:yc+self.roi[2], xc-self.roi[0]:xc+self.roi[0]]

        # 霍夫变换检测直线
        lines = cv2.HoughLines(img_roi, 4.5, np.pi/180, 70, 40)
        if lines is None or len(lines)<2:
            self.board_exist = 0
            self.pos_bias = 0
            self.ang_bias = 0
            return
        lines = lines[:,0,:]

        # 根据rho的绝对值，分成左右两侧
        lines_dist = np.abs(lines[:,0])
        sort_idx = np.argsort(lines_dist) # 按照 dist 排序
        lines = lines[sort_idx]
        lines_dist = lines_dist[sort_idx]
        d_dist = lines_dist[1:] - lines_dist[:-1] # (len-1,)
        mid_idx = np.argmax(d_dist) + 1 # 找最大rho增量对应的idx作为分割点
        if d_dist[mid_idx-1] < 30: # 判断：两堆之间差异不大
            self.board_exist = 0
            self.pos_bias = 0
            self.ang_bias = 0
            return

        #--- 可以分为左右两侧，接下来计算偏差
        self.board_exist = 1
        # 计算两堆线的角平分线
        self.line1 = self.meanline(lines[:mid_idx])
        self.line2 = self.meanline(lines[mid_idx:])

        # 左右两条线的锐角平分线
        self.line_mid = self.meanline(np.vstack([self.line1, self.line2]))
        rho, theta = self.line_mid

        # 计算中心点到两直线的平分线的距离 pos_bias
        c_pt = (self.halfwidth, self.forward) # (x, y)
        # 直线方程 sin(theta)y + cos(theta)x - rho = 0
        # 乘以 +-1 使得x前系数为正，这样代入 c_pt 后计算出的值能反应出偏差
        # 正表示，c_pt 在线右边，负表示在左边
        cos_th = np.cos(theta)
        if cos_th == 0:  # 直线水平
            self.board_exist = 0
            return
        else:
            sign = 1 if cos_th > 0 else -1
            self.pos_bias = sign * ( np.sin(theta)*c_pt[1] + np.cos(theta)*c_pt[0] - rho )

        # 计算角度偏差 = line_mid的角度, 将line_mid的角度映射到 -pi/2 ~ pi/2
        if theta > np.pi/2:
            theta = theta - np.pi
        self.ang_bias = -theta


    def drawline(self, img, line, color, linewidth=2):
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1,y1), (x2,y2), color, linewidth)


    def meanline(self, lines):
        """ 计算线的锐角平分线 """
        #--- 判断线簇中，是否有夹角为钝角，若有则反向
        n = len(lines)
        sort_idx = np.argsort(lines[:, 1]) # 按照角度排序
        min_angle = lines[sort_idx[0], 1]
        inv_idx =  (lines[:, 1] - min_angle > np.pi/2).nonzero()[0] # 夹角为钝角的idx
        lines[inv_idx, 1] -= np.pi              # 将这些射线进行反向
        lines[inv_idx, 0] = -lines[inv_idx, 0]
        #--- 计算锐角角平分线
        mline = np.mean(lines, axis=0, keepdims=False)
        # 平均完后，可能会出现 <0 的情况，规范化到 0~pi
        if mline[1] < 0:
            mline[1] += np.pi
            mline[0] = -mline[0]

        return mline

    def visulize(self):
        """ 返回一张可视化图片 """
        xc, yc = self.center
        img_show = np.dstack((self.lidar_img, self.lidar_img, self.lidar_img)) # 转三通道
        img_roi = img_show[yc-self.roi[1]:yc+self.roi[2], xc-self.roi[0]:xc+self.roi[0]]  # 绘图区域
        cv2.rectangle(img_show, (xc-self.roi[0], yc-self.roi[1]),
                                (xc+self.roi[0], yc+self.roi[2]), (0, 255, 0), 2)

        self.drawline(img_roi, self.line1, (0,0,255))
        self.drawline(img_roi, self.line2, (255,0,0))
        self.drawline(img_roi, self.line_mid, (255,255,255))
        font = cv2.FONT_HERSHEY_SIMPLEX
        center_text = "pos_bias:%.2fcm | ang_bias:%.2f " % (self.pos_bias, rad2deg(self.ang_bias))
        cv2.putText(img_show, center_text, (50, 50), font, 0.5, (20, 20, 255), 2)
        return img_show



if __name__ == '__main__':
    Lidar = lidar_node()
    while not rospy.is_shutdown():
        img = Lidar.visulize()
        cv2.imshow('lidar', img)
        cv2.waitKey(5)
