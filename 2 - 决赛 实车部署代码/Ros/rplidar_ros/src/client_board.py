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

imSize = (520, 520)
flag = 0
bias = 0
angle = 0
isShow = True

""" sub topics """
topic_scan = "/scan"
def callback_scan(scan):
    img = np.zeros(imSize, dtype=np.uint8)
    ranges = np.array(scan.ranges)  # [1440,]
    ranges[ranges>2.5] = 0
    angle = np.arange(len(scan.ranges)) * scan.angle_increment
    px = ( np.cos(angle) * ranges * 100 ).astype(np.int32) + 260
    py = ( np.sin(angle) * ranges * 100 ).astype(np.int32) + 260
    img[px, py] = 255
    if isShow:
        img_show = process(img)
        cv2.imshow('scan', img_show)
        cv2.waitKey(5)


def process(img):

    global flag
    global bias
    global angle

    img_show = np.dstack((img, img, img))
    # 剪裁
    img_roi = img[260-80:260+20, 260-80:260+80]
    img_show_roi = img_show[260-80:260+20, 260-80:260+80]
    cv2.rectangle(img_show, (170, 170), (350, 290), (0, 255, 0), 2)

    # 霍夫变换检测直线
    lines = cv2.HoughLines(img_roi, 4.5, np.pi/180, 70, 40)
    if lines is None:
        return img_show

    lines = lines[:,0,:]
    if len(lines)<2: # 判断：线数少于2
        print('线数<2')
        flag = 0
        return img_show

    # 根据rho的绝对值，分成左右两侧
    lines_dist = np.abs(lines[:,0])
    sort_idx = np.argsort(lines_dist) # 按照 dist 排序
    lines = lines[sort_idx]
    lines_dist = lines_dist[sort_idx]
    d_dist = lines_dist[1:] - lines_dist[:-1] # (len-1,)
    mid_idx = np.argmax(d_dist) + 1 # 找最大rho增量对应的idx作为分割点
    if d_dist[mid_idx-1] < 30: # 判断：两堆之间差异不大
        print('分不出两份')
        flag = 0
        return img_show

    # 计算两堆线的角平分线
    line1 = meanline(lines[:mid_idx])
    line2 = meanline(lines[mid_idx:])

    # 可视化
    if isShow:
        drawline(img_show_roi, line1, (0,0,255))
        drawline(img_show_roi, line2, (255,0,0))
        print(lines)

    #--- 左右两条线的锐角平分线
    line_mid = meanline(np.vstack([line1, line2]))
    rho, theta = line_mid

    #--- 计算中心点到两直线的平分线的距离 pos_bias
    c_pt = (80, 80) # (x, y)
    # 直线方程 sin(theta)y + cos(theta)x - rho = 0
    # 乘以 +-1 使得x前系数为正，这样代入 c_pt 后计算出的值能反应出偏差
    # 正表示，c_pt 在线右边，负表示在左边
    cos_th = np.cos(theta)
    if cos_th == 0:  # 直线水平
        pos_bias = 65535  # 表示障碍物
    else:
        sign = 1 if cos_th > 0 else -1
        pos_bias = sign * ( np.sin(theta)*c_pt[1] + np.cos(theta)*c_pt[0] - rho )

    #--- 计算角度偏差 = line_mid的角度
    # 将line_mid的角度映射到 -pi/2 ~ pi/2
    if theta > np.pi/2:
        theta = theta - np.pi
    ang_bias = theta
    print('偏差 pos_bias=%5.2f, ang_bias=%5.2f' % (pos_bias, ang_bias))
    return img_show


def drawline(img, line, color, linewidth=2):
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


def meanline(lines):
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


def main():

    # --- node init
    rospy.init_node('lidar_board', anonymous=True)
    print("[lidar_board]: Init")

    # --- publisher topic
    board_detection_pub = rospy.Publisher('/lida/board_detect', BoardMsg, queue_size=100)
    rate = rospy.Rate(10)
    # --- subscriber topic
    rospy.Subscriber(topic_scan, LaserScan, callback_scan)
    # --- subscriber thread
    thread_spin = threading.Thread(target=rospy.spin)
    thread_spin.start()

    while not rospy.is_shutdown():
        # --- subscriber topic
        rate.sleep()
        board_detection_pub.publish(BoardMsg(flag = flag, bias = bias, angle = angle))
        rospy.loginfo("flag = %d, bias = %4.2f, angle = %4.2f", flag, bias, angle)


if __name__ == '__main__':
    main()