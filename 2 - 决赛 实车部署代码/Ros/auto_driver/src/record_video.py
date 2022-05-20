#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
拍摄视频，保存在out_videos/中
"""
import cv2
import datetime
import os
from cap_init import CapInit
import sys
import signal
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = BASE_DIR + '/out_videos'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


def out_name():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.mp4'


if __name__ == '__main__':
    cap = CapInit()

    frame_cnt = 0
    while True:  # 保存1000帧
        ret, frame = cap.read()
        if ret:
            if frame_cnt==0: # 为视频第一帧
                frame_cnt += 1
                out_file = str(SAVE_DIR + '/' + out_name())
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width = frame.shape[:2]
                out_writer = cv2.VideoWriter(out_file, fourcc, 24, (width, height), True)
                print('Start Recording')

                def sigint_handler(signal, frame):
                    """ SIGINT Signal handler """
                    print "Interrupt!"
                    # out_writer.release()
                    print('Save video at: %s' % out_file)
                    sys.exit(0)
                signal.signal(signal.SIGINT, sigint_handler)

            else:
                out_writer.write(frame)
                frame_cnt += 1





