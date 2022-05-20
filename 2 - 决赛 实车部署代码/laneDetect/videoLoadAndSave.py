#coding=utf-8
import os, numpy
import cv2
from tqdm import tqdm
import numpy as np


def read_frame(path):
    # 获得视频的格式
    videoCapture = cv2.VideoCapture(path)

    # 获得码率及尺寸
    # fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_set = []

    # 读帧
    success, frame = videoCapture.read()
    while success:
        # cv2.imshow('windows', frame)  # 显示
        frame_set.append(frame)
        # cv2.waitKey(int(1000 / int(fps)))  # 延迟
        success, frame = videoCapture.read()  # 获取下一帧

    videoCapture.release()

    return success, frame_set


def main():
    path = '..\socket\out_videos\\'
    save_path= '..\socket\out_photos\\'
    file_names = os.listdir(path)
    i = 0
    for file_name in file_names:
        if file_name.endswith('mp4'):
            print(file_name)
            sucess, frame_set = read_frame(path + file_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            for frame in frame_set:
                cv2.imwrite(save_path + '_%05d.jpg' % i, frame)
                i += 1
    # file_name = '00204_h.mp4'

if __name__ == '__main__':
    main()

