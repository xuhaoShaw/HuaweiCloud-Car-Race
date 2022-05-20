#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import sys
import time
import numpy as np
import threading
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32
import cv2

imSize = (520, 520)

stopFlag = 0
distance_obs = 0.7


""" sub topics """
topic_scan = "/scan"
def callback_scan(scan):
    global stopFlag
    global distance_obs
    angle_increment_ = scan.angle_increment * 180 / math.pi


    img = np.zeros(imSize, dtype=np.uint8)
    ranges = np.array(scan.ranges)  # [1440,]
    ranges_obs = ranges
    ranges_obs[ranges_obs>2.5]=0
    
    angle_1 = 150
    angle_2 = 210
    angle_1_num = int((angle_1) / angle_increment_)
    angle_2_num = int((angle_2) / angle_increment_)
    angle_1to2 = np.arange(angle_1_num, angle_2_num) *scan.angle_increment
    ranges_1to2 = ranges_obs[angle_1_num:angle_2_num]
    #print(angle_increment_)
    #print(angle_1to2)

    px = ( np.cos(angle_1to2) * ranges_1to2 * 100 ).astype(np.int32) + 260
    py = ( np.sin(angle_1to2) * ranges_1to2 * 100 ).astype(np.int32) + 260
    img2 = np.zeros(imSize, dtype=np.uint8)    
    img2[px, py] = 255

    ranges_1to2[ranges_1to2>distance_obs] = 0
    y = np.nonzero(ranges_1to2)
    if len(y[0])>10:
        stopFlag = 1
    else:
        stopFlag = 0

    cv2.imshow('scan2', img2)
    cv2.waitKey(5)


def main():
    #--- node init
    rospy.init_node('lidar_port', anonymous=True)
    print("[lidar_port]: Init")

    
    #--- publisher topic
    obstacle_detection_pub = rospy.Publisher('/lida/is_obstacle', Int32, queue_size = 100)
    rate = rospy.Rate(5)
    rospy.Subscriber(topic_scan, LaserScan, callback_scan)
    thread_spin = threading.Thread(target = rospy.spin)
    thread_spin.start()

    while not rospy.is_shutdown():
        #--- subscriber topic
        
        rate.sleep()
        lidar_msg = Int32()
        lidar_msg = stopFlag

        obstacle_detection_pub.publish(lidar_msg)

        rospy.loginfo("publish:前方60°范围，%fm内障碍物信息 %d", distance_obs, lidar_msg)

if __name__ == '__main__':
    main()


