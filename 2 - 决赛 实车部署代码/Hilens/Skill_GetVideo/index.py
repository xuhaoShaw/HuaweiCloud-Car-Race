#! /usr/bin/python3.7
# -*- coding: utf-8 -*-

#本例子直接拉取摄像头画面帧，然后在上面打印Hollo World，最后输出到显示器
import hilens
import cv2
import numpy as np
from npsocket import NumpySocket


def run():

    # socket
    sock_sender = NumpySocket()
    sock_sender.initialize_sender('192.168.43.215', 7777)

    # 构造摄像头
    cap = hilens.VideoCapture()
    #disp = hilens.Display(hilens.HDMI)

    while True:

        # 获取一帧画面
        frame = cap.read()

        # 使用opencv进行颜色转换
        bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21)
        bgr = cv2.resize(bgr, (640, 360))

        # 使用hilens进行颜色转换
        nv21 = hilens.cvt_color(bgr, hilens.BGR2YUV_NV21)

        #输出到HDMI
        #disp.show(nv21)

        #输出到socket
        sock_sender.send_array(bgr)


if __name__ == '__main__':
    hilens.init("hello")

    run()

    hilens.terminate()