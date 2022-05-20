#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
import time
# ms_time  = lambda: (int(round(time.time() * 1000)))

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# 常量定义
# 摄像头
frameWidth = 1280  # 宽
frameHeight = 720  # 长
frameFps = 30  # 帧率
camMat = np.array([[6.678151103217834e+02, 0, 6.430528691213178e+02],
                   [0, 7.148758960098705e+02, 3.581815819255082e+02], [0, 0, 1]])  # 相机校正矩阵
camDistortion = np.array([[-0.056882894892153, 0.002184364631645, -0.002836821379133, 0, 0]])  # 相机失真矩阵


# 透视变换
src_points = np.array([[0., 527.], [416., 419.], [781., 420.], [1065., 542.]], dtype="float32")  # 源点
dst_points = np.array([[266., 686.], [266., 19.], [931., 20.], [931., 701.]], dtype="float32")  # 目标点
MWarp = cv2.getPerspectiveTransform(src_points, dst_points)  # 透视变换矩阵计算


# 视觉处理
kerSz = (3, 3)  # 膨胀与腐蚀核大小
grayThr = 145  # 二值化阈值
roiXRatio = 3/5  # 统计x方向上histogram时选取的y轴坐标范围，以下方底边为起始点，比例定义终止位置
roiXBase = 0.3  # 统计左右初始窗的y轴范围
nwindows = 15  # 窗的数目
window_width = 200  # 窗的宽度
minpix = 250  # 最小连续像素，小于该长度的被舍弃以去除噪声影响


# 距离映射
x_cmPerPixel = 90 / 665.00  # x方向上一个像素对应的真实距离 单位：cm
y_cmPerPixel = 81 / 680.00  # y方向上一个像素对应的真实距离 单位：cm
roadWidth = 80  # 道路宽度 单位：cm
y_offset = 50.0  # 由于相机位置较低，识别到的车道线距离车身较远，不是当前位置，定义到的车道线与车身距离 单位：cm<no usage>
cam_offset = 18.0  # 相机中心与车身中轴线的距离 单位：cm


# 控制
I = 58.0  # 轴间距<no usage>
k = -19  # 计算cmdSteer的系数<no usage>

""" 单帧处理过程
图像预处理
生成基点：
    第一帧，计算左右基点
    从第二帧开始，从上一帧继承基点
currentx = 基点
迭代，求出每个窗中车道线中心点：
    生成窗
    更新 currentx:
        if 窗中白点>minpix, 为检测到车道：
            currentx = 统计窗中白色部分 x 平均值 xc_lane
        elif 若右/左侧检测到车道线：
            左/右车道线根据右/左车道线更新
        elif 两侧都没检测到：
            不更新，沿用上一窗中心值
        (TODO 更好的更新方法也许是用前面获得的点拟合出下一点)
        记录xc_lane用于拟合
    可视化

"""

class camera:
    def __init__(self):
        self.camMat = camMat   # 相机校正矩阵
        self.camDistortion = camDistortion  # 相机失真矩阵
        self.cap = cv2.VideoCapture(str(BASE_DIR / 'challenge_video.mp4'))  # 读入视频
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)  # 设置读入图像宽
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)  # 设置读入图像长
        self.cap.set(cv2.CAP_PROP_FPS, frameFps)  # 设置读入帧率
        self.kernal = np.ones(kerSz, np.uint8)  # 定义膨胀与腐蚀的核
        self.l_lane_centers = np.zeros((nwindows, 2)).astype(np.int32) # 左车道线中心点，用于拟合
        self.r_lane_centers = np.zeros((nwindows, 2)).astype(np.int32) # 右车道线中心点
        self.win_width = window_width
        self.win_height = frameHeight * roiXRatio // nwindows
        self.n_win = nwindows

    def __del__(self):
        self.cap.release()  # 释放摄像头

    def spin(self):
        ret, img = self.cap.read()  # 读入图片
        if ret == True:
            #--- 校正，二值化，透视变化
            binary_warped = self.preprocess(img)
            binary_show = binary_warped.copy()
            self.win_height = int(binary_warped.shape[0] * roiXRatio / nwindows)  # 窗的高度
            h, w = binary_warped.shape[:2]
            #--- 生成左, 右基点
            if self.l_lane_centers[0,1]==0: # 第一帧，通过统计生成，要求左右车道线都在视野内
                histogram_x = np.sum(binary_warped[int(binary_warped.shape[0] * (1-roiXBase)):, :], axis=0)  # 计算 x方向直方图 [x,]
                midpoint = int(histogram_x.shape[0] / 2)                        # x方向中点，用来判断左右
                left_base = np.argmax(histogram_x[:midpoint])                   # 定义左车道线的基点
                right_base = np.argmax(histogram_x[midpoint:]) + midpoint       # 定义右车道线的基点
            else:
                left_base = self.l_lane_centers[0, 0]    # 上一帧第一个窗中心的x坐标
                right_base = self.r_lane_centers[0, 0]   # 上一帧第一个窗中心的y坐标

            # 绘制初始窗基点
            cv2.circle(binary_show, (left_base, h-10), 4, 125, -1)
            cv2.circle(binary_show, (right_base, h-10), 4, 125, -1)
            cv2.line(binary_show, (w//2, 0), (w//2, h-1), 127, 4)  # 中线

            #--- 初始化, 开始迭代所有窗, 求出每个窗中车道线中心点
            left_current = left_base    # 左车道线当前窗x中心位置 设为基点
            right_current = right_base  # 右车道线当前窗x中心位置
            for i in range(nwindows):
                win_yc = int(h - (i + 0.5) * self.win_height)
                win_left = self.get_win(binary_warped, xc=left_current, yc=win_yc)  # 左窗
                win_right = self.get_win(binary_warped, xc=right_current, yc=win_yc) # 右窗

                cv2.rectangle(  binary_show,
                                (int(left_current-self.win_width/2), int(win_yc-self.win_height/2)),
                                (int(left_current+self.win_width/2), int(win_yc+self.win_height/2)), 255, 2)  # 在图中画出左车道线的窗
                cv2.rectangle(  binary_show,
                                (int(right_current-self.win_width/2), int(win_yc-self.win_height/2)),
                                (int(right_current+self.win_width/2), int(win_yc+self.win_height/2)), 255, 2)  # 在图中画出右车道线的窗

                good_left_x = win_left.nonzero()[1]
                good_right_x = win_right.nonzero()[1]
                # print(win_left.shape[0]*win_left.shape[1], len(good_left_x), len(good_left_x))
                # 若检测到车道线，用平均值更新中点，否则，不更新 TODO：拟合出下一个点
                if len(good_left_x) > minpix:
                    left_current = int(left_current + np.mean(good_left_x) - self.win_width/2)  # 更新左车道线窗的x中心位置
                if len(good_right_x) > minpix:
                    right_current = int(right_current + np.mean(good_right_x) - self.win_width/2)  # 更新右车道线窗的x中心位置
                if i > 0:
                    if len(good_left_x)>minpix and len(good_right_x)<minpix: # 如果左侧检测到车道线，用右侧近似更新右侧
                        left_dx = left_current - self.l_lane_centers[i-1, 0]
                        right_current += left_dx
                    elif len(good_right_x)>minpix and len(good_left_x)<minpix: # 如果右侧检测到车道线，用左侧近似更新右侧
                        right_dx = right_current - self.r_lane_centers[i-1, 0]
                        left_current += right_dx

                # 记录检测到的中心点
                self.l_lane_centers[i, :] = [left_current, win_yc] # 记录左车道线窗的中点 cx, cy
                self.r_lane_centers[i, :] = [right_current, win_yc] # 右车道线窗的中点 cx, cy

                # 可视化，画出窗
                cv2.circle(binary_show, (left_current, win_yc), 4, 125, -1)
                cv2.circle(binary_show, (right_current, win_yc), 4, 125, -1)

            cv2.imshow('binary_show', binary_show)  # 显示每一帧窗的位置
            cv2.waitKey(1)

            #--- 拟合
            left_fit = np.polyfit(self.l_lane_centers[:,1], self.l_lane_centers[:,0], 2)  # 左车道拟合
            right_fit = np.polyfit(self.r_lane_centers[:,1], self.r_lane_centers[:,0], 2)  # 右车道拟合
            ymax = binary_warped.shape[0]-1
            y = np.linspace(0, ymax, ymax+1)  # 定义自变量 y
            leftx_fit = np.polyval(left_fit, y)  # 计算拟合后左车道线的x坐标
            rightx_fit = np.polyval(right_fit, y)  # 计算拟合后右车道线的x坐标
            left_fit_real = np.polyfit(y * y_cmPerPixel, leftx_fit * x_cmPerPixel, 2)  # 映射到现实尺度下左车道线的拟合
            right_fit_real = np.polyfit(y * y_cmPerPixel, rightx_fit * x_cmPerPixel, 2)  # 映射到现实尺度下右车道线的拟合
            if np.absolute(2*left_fit_real[0])==0 or np.absolute(2*right_fit_real[0])==0:  # 壁免除零
                left_curverad = 1000
                right_curverad = 1000
            else:
                left_curverad = ((1 + (2*left_fit_real[0]*ymax*y_cmPerPixel + left_fit_real[1])**2)**1.5)\
                                / np.absolute(2*left_fit_real[0])  # 左车道线曲率半径
                right_curverad = ((1 + (2*right_fit_real[0]*ymax*y_cmPerPixel + right_fit_real[1])**2)**1.5)\
                             / np.absolute(2*right_fit_real[0])  # 右车道线曲率半径
            curverad = (left_curverad + right_curverad) / 2  # 整体曲率半径
            lane_width = np.absolute(leftx_fit[ymax] - rightx_fit[ymax])  # 车道线的像素宽度
            lane_cmPerPixel = roadWidth / lane_width  # 车道线的像素比例
            cen_pos = ((leftx_fit[ymax] + rightx_fit[ymax]) * lane_cmPerPixel) / 2.0  # 车道中心线位置
            veh_pos = binary_warped.shape[1] * lane_cmPerPixel / 2.0  # 小车位置，目前定义为画面中心，但是摄像头与小车中轴线不一定重合，需要校准
            distance_from_center = veh_pos - cen_pos  # 离中心距离，<0位于左边, >0位于右边

            # 绘图显示
            color_warp = np.zeros_like(img).astype(np.uint8)
            pts_left = np.transpose(np.vstack([leftx_fit, y])).astype(np.int32)
            pts_right = np.flipud(np.transpose(np.vstack([rightx_fit, y]))).astype(np.int32)
            pts = np.vstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, [pts,], (0, 255, 0))
            cv2.imshow('result1', color_warp)
            cv2.waitKey(1)
            newwarp = cv2.warpPerspective(color_warp, MWarp, (img.shape[1], img.shape[0]), None, cv2.WARP_INVERSE_MAP)
            result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            radius_text = "Radius of Curvature: %scm" % (round(curverad))
            cv2.putText(result, radius_text, (100, 100), font, 1, (20, 20, 255), 2)
            pos_flag = 'right' if distance_from_center>0 else 'left'
            center_text = "Vehicle is %.3fcm %s of center" % (abs(distance_from_center), pos_flag)
            cv2.putText(result, center_text, (100, 150), font, 1, (20, 20, 255), 2)
            cv2.imshow('result', result)
            cv2.waitKey(1)

    def preprocess(self, img):
        """
        取下方区域，矫正畸变，二值化，透视变换
        """
        mask = np.zeros_like(img)  # 创建遮罩
        cv2.rectangle(mask, (0, int(img.shape[0] * (1 - roiXRatio))), (img.shape[1], img.shape[0]), (255, 255, 255), cv2.FILLED)  # 填充遮罩
        segment = cv2.bitwise_and(img, mask)  # 取出遮罩范围
        undist_img = cv2.undistort(segment, self.camMat, self.camDistortion, None, self.camMat)  # 校正畸变图像
        # gray_Blur = cv2.dilate(gray_Blur, self.kernel, iterations = 1)  # 膨胀
        gray_Blur = cv2.erode(undist_img, self.kernal, iterations=1)  # 腐蚀
        _, gray_img = cv2.threshold(gray_Blur, grayThr, 255, cv2.THRESH_BINARY) # 二值化
        gray_img = np.mean(gray_img, axis=2).astype(np.uint8)  # 单通道化
        perspect_img = cv2.warpPerspective(gray_img, MWarp, (gray_Blur.shape[1], gray_Blur.shape[0]),
                                            cv2.INTER_LINEAR)  # 透视变换
        return perspect_img


    def get_win(self, img, xc, yc):
        """
        从图中取出一个窗, xc, yc 为窗中心点
        """
        ymax, xmax = img.shape
        half_w = self.win_width // 2
        half_h = self.win_height // 2
        ylow = max(yc-half_h, 0)
        yhigh = min(yc+half_h, ymax)
        xlow = min(max(xc-half_w, 0), xmax)
        xhigh = max(min(xc+half_w, xmax), 0)
        return img[ylow:yhigh, xlow:xhigh]



if __name__ == '__main__':
    try:
        cam = camera()
        while True:
            cam.spin()
    except:
        print("helloworld")
        pass


