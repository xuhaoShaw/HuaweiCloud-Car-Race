#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import rospy
import time
from npsocket import NumpySocket
from bluetooth_bridge.msg import HilensMsg

sock_receiver = NumpySocket()
sock_receiver.initalize_receiver('192.168.2.1', 9999)

topic_hilens = '/hilens'

if __name__ == '__main__':
    rospy.init_node('hilens_receiver', anonymous=False)
    print("[Hilens Node]: Init")
    pub_hilens = rospy.Publisher(topic_hilens, HilensMsg, queue_size=10)
    time.sleep(0.5)

    while not rospy.is_shutdown():
        conn, bboxes = sock_receiver.receive_array()
        if conn:
            msg = HilensMsg()
            for bbox in bboxes:
                label = int(bbox[4])
                if label != -1:
                    msg.flag[label] = 1
                    msg.x_min[label] = int(bbox[0])
                    msg.y_min[label] = int(bbox[1])
                    msg.x_max[label] = int(bbox[2])
                    msg.y_max[label] = int(bbox[3])
                    msg.score[label] = bbox[5]
            pub_hilens.publish(msg)

