#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
搜索用 cv2.inRange 选取颜色范围的最佳参数
"""

import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
import cv2

img = cv2.imread(str(BASE_DIR/'video/hsv.jpg'), cv2.IMREAD_COLOR)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将BGR图像转换为HSV格式


idx = 0
hsv_name  = ['hl', 'sl', 'vl', 'hh', 'sh', 'vh']
hsv_range = [ 19, 125, 145, 77, 254, 255]
while True:
    lower_color = np.array(hsv_range[:3])  # 分别对应着HSV中的最小值
    upper_color = np.array(hsv_range[3:])  # 分别对应着HSV中的最大值
    cv2.imshow('origin', img)

    mask = cv2.inRange(img_hsv, lower_color, upper_color)  # inrange函数将根据最小最大HSV值检测出自己想要的颜色部分
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "hsv low : {:d}, {:d}, {:d} | high: {:d}, {:d}, {:d} ".format(*hsv_range)
    text1 = 'push "j" or "k" to change: ' + hsv_name[idx]
    cv2.putText(mask, text, (100, 50), font, 1, 255, 2)
    cv2.putText(mask, text1, (100, 100), font, 1, 255, 2)
    cv2.imshow("mask", mask)  # 通过imshow显示
    c = cv2.waitKey(0)  # 等待键盘按下按键退出
    if c == 27: # 按下esc退出
        print(lower_color, upper_color)
        break
    if c in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]: # 更改那个变量
        idx = int(chr(c)) - 1
        print(f'Change {hsv_name[idx]}')
    if c in [ord('j'), ord('j')]:
        hsv_range[idx] = min(hsv_range[idx]+1, 255)
    if c in [ord('k'), ord('K')]:
        hsv_range[idx] = max(hsv_range[idx]-1, 0)

