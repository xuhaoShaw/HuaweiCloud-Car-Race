#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
import cv2
import numpy as np
import time
# ms_time  = lambda: (int(round(time.time() * 1000)))


class laneDetect:
    """ 车道线检测类 """
    def __init__(self, Mwarp, camMat, camDistortion, kerSz, grayThr, frameHeight, frameWidth, roiXRatio,
                winWidth, winNum, winThr, pixThr, roadWidCm, roadWidPix, isShow):
        self.Mwarp  = Mwarp                 # 透视变换矩阵
        self.camMat = camMat                # 相机校正矩阵
        self.camDistortion = camDistortion  # 相机失真矩阵
        self.kernal = np.ones(kerSz, np.uint8)      # 定义膨胀与腐蚀的核
        self.gray_thr = grayThr                     # 二值化阈值
        self.frame_h = frameHeight
        self.frame_w = frameWidth
        self.roiXRatio = roiXRatio                  # 遮罩保留的y轴坐标范围，以下方底边为起始点，比例定义终止位置
        self.win_w = winWidth                       # 窗宽
        self.win_h = int(self.frame_h // winNum)    # 窗高
        self.win_n = winNum                         # 窗数
        self.show = isShow                          # 是否绘图显示
        self.road_w_pix = roadWidPix                # 车道宽（像素,默认700）
        self.road_w_cm  = roadWidCm                 # 道路宽度 单位：cm
        self.cm_per_pix = roadWidCm / roadWidPix    # 车道线内部一个像素对应的真实距离 单位：cm
        self.wins_thr = winThr                      # 车道线检出需满足 检出窗数 > wins_thr
        self.pix_thr = pixThr                        # 一个窗中检出车道线需满足 非零像素 > minpix
        self.lane_xc = np.zeros((2, self.win_n)).astype(np.int32)  # 左右车道线中心点x坐标，lane_center_x[0]为左，[1]为右
        self.lane_xc[1, :] = frameWidth-1
        self.lane_yc = np.arange(int(self.frame_h - 0.5*self.win_h), 0, -self.win_h)  # 车道线中心点 y 坐标
        self.lane_flag = np.full((2, self.win_n), False, np.bool_)      # 左右车道线 检出标志位
        self.lane_curve = [None, None]                               # 左右拟合曲线
        self.bias = 0           # veh_pos - cen_pos, >0偏右，<0偏左，cm

    def spin(self, img):
        #--- 校正，二值化，透视变化
        img_prep = self.(img)
        if self.show:
            img_show = np.repeat(img_prep[:, :, None], 3, axis=2)

        #--- 迭代更新 lane_xc
        for i in range(self.win_n):
            # 窗中心
            win_yc = self.lane_yc[i]                                # 第i层左右车道预测窗中心 y 坐标
            l_win_xc, r_win_xc = self.check_order(*self.porpose_win_xc(img_prep, i))      # 第i层左右车道预测窗中心 x 坐标
            # l_win_xc, r_win_xc = self.check_order2(*self.porpose_win_xc(img_prep, i), img_prep, win_yc) # 第i层左右车道预测窗中心 x 坐标
            # 生成窗
            l_win_pts = self.get_win(img_prep, xc=l_win_xc, yc=win_yc)  # 左窗中所有像素点坐标 [n,2]
            r_win_pts = self.get_win(img_prep, xc=r_win_xc, yc=win_yc)  # 右窗
            # 绘制窗
            if self.show:
                cv2.rectangle(  img_show,
                                (int(l_win_xc-self.win_w/2), int(win_yc-self.win_h/2)),
                                (int(l_win_xc+self.win_w/2), int(win_yc+self.win_h/2)), (127,127,255), 2)  # 在图中画出左车道线的窗
                cv2.rectangle(  img_show,
                                (int(r_win_xc-self.win_w/2), int(win_yc-self.win_h/2)),
                                (int(r_win_xc+self.win_w/2), int(win_yc+self.win_h/2)), (255,127,127), 2)  # 在图中画出右车道线的窗

            # 检测窗中的车道线中心 lane_xc
            l_det = len(l_win_pts) > self.pix_thr   # 检测到左车道线中点
            r_det = len(r_win_pts) > self.pix_thr   # 检测到右车道线中点

            if l_det:
                self.lane_xc[0, i] = int(np.mean(l_win_pts))  # 更新左车道线的 x 中心位置
            else:
                self.lane_xc[0, i] = l_win_xc
            if r_det:
                self.lane_xc[1, i] = int(np.mean(r_win_pts))  # 更新右车道线的 x 中心位置
            else:
                self.lane_xc[1, i] = r_win_xc
            self.lane_flag[:, i] = [l_det, r_det]   # 更新检出标志位

        #--- 拟合
        l_win_nums = np.sum(self.lane_flag[0]) # 有效窗口数
        r_win_nums = np.sum(self.lane_flag[1])
        if l_win_nums > 2: #  如果检出点数>2就用检出点进行拟合，否则用所有检出点+预测窗中心拟合
            l_curve = np.polyfit(self.lane_yc[self.lane_flag[0]], self.lane_xc[0, self.lane_flag[0]], 2)   # 左车道线拟合
        else:
            l_curve = np.polyfit(self.lane_yc, self.lane_xc[0], 2)
        if r_win_nums > 2:
            r_curve = np.polyfit(self.lane_yc[self.lane_flag[1]], self.lane_xc[1, self.lane_flag[1]], 2)   # 左车道线拟合
        else:
            r_curve = np.polyfit(self.lane_yc, self.lane_xc[1], 2)
        self.update_curve(l_curve, 0)   # 更新左，右车道拟合线，过滤突变
        self.update_curve(r_curve, 1)

        # 计算航向偏差
        if l_win_nums>=self.wins_thr and r_win_nums>=self.wins_thr:
            lx = np.polyval(self.lane_curve[0], self.frame_h)
            rx = np.polyval(self.lane_curve[1], self.frame_h)
            cen_pos = (lx + rx) / 2.0       # 车道中心线位置
            veh_pos = self.frame_w / 2.0    # 小车位置，目前定义为画面中心
            self.bias = (veh_pos - cen_pos) * self.cm_per_pix
        elif l_win_nums >= r_win_nums and l_win_nums>0:  # 只检出左车道线
            cen_pos = np.polyval(self.lane_curve[0], self.frame_h)*self.cm_per_pix + self.road_w_cm / 2  # 车道中心线位置
            veh_pos = self.frame_w / 2.0  * self.cm_per_pix # 小车位置，目前定义为画面中心
            self.bias = veh_pos - cen_pos
        elif r_win_nums > l_win_nums:   # 只检出右车道线
            cen_pos = np.polyval(self.lane_curve[1], self.frame_h)*self.cm_per_pix - self.road_w_cm / 2  # 车道中心线位置
            veh_pos = self.frame_w / 2.0  * self.cm_per_pix # 小车位置，目前定义为画面中心
            self.bias = veh_pos - cen_pos

        #--- 绘图
        if self.show:
            # (1) 绘制车道点
            xc_idxs = self.lane_flag.nonzero()
            xcs = self.lane_xc[xc_idxs]
            for i in range(len(xcs)):
                point = (xcs[i], self.lane_yc[xc_idxs[1][i]])
                cv2.circle(img_show, point, 4, (125,125,255), -1)

            # (2) 绘制拟合车道线
            ymax = self.frame_h - 1
            y = np.arange(0, ymax, 2)  # 定义自变量 y
            if  l_win_nums >= self.wins_thr and r_win_nums >= self.wins_thr:
                l_x = np.polyval(self.lane_curve[0], y)
                r_x = np.polyval(self.lane_curve[1], y)
                pts_left = np.transpose(np.vstack([l_x, y])).astype(np.int32)
                pts_right = np.flipud(np.transpose(np.vstack([r_x, y]))).astype(np.int32)
                pts = np.vstack((pts_left, pts_right))  # 拟合点
            elif l_win_nums >= r_win_nums and l_win_nums>0:
                l_x = np.polyval(self.lane_curve[0], y)
                pts = np.transpose(np.vstack([l_x, y])).astype(np.int32)
            elif l_win_nums < r_win_nums:
                r_x = np.polyval(self.lane_curve[1], y)
                pts = np.flipud(np.transpose(np.vstack([r_x, y]))).astype(np.int32)
            else:
                pts = np.array([[0, 0]]).astype(np.int32)
            cv2.polylines(img_show, [pts,], False, (0, 255, 0), 2)
            #cv2.namedWindow('img_show', cv2.WINDOW_NORMAL)
            #cv2.imshow('img_show', img_show)
            #cv2.waitKey(1)

            # (3) 映射到真实图像, 绘图
            color_warp = np.zeros_like(img).astype(np.uint8)
            cv2.fillPoly(color_warp, [pts,], (0, 255, 0))
            newwarp = cv2.warpPerspective(color_warp, self.Mwarp, (self.frame_w, self.frame_h), None, cv2.WARP_INVERSE_MAP)
            result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            side_text = 'right' if self.bias>0 else 'left'
            center_text = "Vehicle is %.3f cm %s of center" % (abs(self.bias), side_text)
            cv2.putText(result, center_text, (50, 50), font, 1, (20, 20, 255), 2)
            #cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            #cv2.imshow('result', result)
            #cv2.waitKey(1)
            #c = cv2.waitKey(0)
            #if c == 27:
            #    exit()
            return self.bias, result

    def preprocess(self, img):
        """
        取下方区域，矫正畸变，二值化，透视变换
        """
        mask = np.zeros_like(img)  # 创建遮罩
        cv2.rectangle(mask, (0, int(img.shape[0] * (1 - self.roiXRatio))), (img.shape[1], img.shape[0]), (255, 255, 255), cv2.FILLED)  # 填充遮罩
        segment = cv2.bitwise_and(img, mask)  # 取出遮罩范围
        undist_img = cv2.undistort(segment, self.camMat, self.camDistortion, None, self.camMat)  # 校正畸变图像
        gray_Blur = cv2.dilate(undist_img, self.kernal, iterations = 1)  # 膨胀
        # gray_Blur = cv2.erode(undist_img, self.kernal, iterations=1)  # 腐蚀
        _, gray_img = cv2.threshold(gray_Blur, self.gray_thr, 255, cv2.THRESH_BINARY) # 二值化
        gray_img_1c = np.mean(gray_img, axis=2).astype(np.uint8)  # 单通道化
        perspect_img = cv2.warpPerspective(gray_img_1c, self.Mwarp, (gray_Blur.shape[1], gray_Blur.shape[0]),
                                            cv2.INTER_LINEAR)  # 透视变换
        return perspect_img


    def get_win(self, img, xc, yc):
        """
        从图中取出一个窗中所有像素点的x位置, [n, 1(x, )]
        """
        half_w = self.win_w // 2
        half_h = self.win_h // 2
        ylow = max(yc-half_h, 0)
        yhigh = min(yc+half_h, self.frame_h)
        xlow = min(max(xc-half_w, 0), self.frame_w)
        xhigh = max(min(xc+half_w, self.frame_w), 0)
        win = img[ylow:yhigh, xlow:xhigh]
        good_x = win.nonzero()[1] + xc - self.win_w//2  # 非零像素 x 坐标
        return good_x

    def check_order(self, l_xc, r_xc):
        """ 对 porpose_win_xc 得到的窗中心进行检查，避免两窗过于接近 """
        if (r_xc-l_xc)<50:
            xc_mean = np.mean(self.lane_xc, axis=1)  # 窗xc的平均值
            error = np.abs(xc_mean - [l_xc, r_xc])
            if error[0] < error[1]:
                r_xc = l_xc + self.road_w_pix
            else:
                l_xc = r_xc - self.road_w_pix
        return l_xc, r_xc

    def check_order2(self, l_xc, r_xc, img, yc):
        """ 对 porpose_win_xc 得到的窗中心进行检查 """
        xc_mean = np.mean(self.lane_xc, axis=1)  # 窗xc的平均值
        error = xc_mean - [l_xc, r_xc]
        # 避免左窗在右车道线，右窗在更右侧或相反情况（检查左车道线左侧和右车道线右侧）
        if l_xc > self.frame_w//2:
            win_pts = self.get_win(img, l_xc-self.road_w_pix, yc)
            if len(win_pts) > self.pix_thr:
                r_xc = l_xc
                l_xc = l_xc-self.road_w_pix
        elif r_xc < self.frame_w//2:
            win_pts = self.get_win(img, r_xc+self.road_w_pix, yc)
            if len(win_pts) > self.pix_thr:
                l_xc = r_xc
                r_xc = r_xc+self.road_w_pix
        # 避免两窗在同一车道线上，或者左窗右窗顺序相反
        if (r_xc-l_xc)<80:
            # error更大的窗为错误窗
            if abs(error[0]) < abs(error[1]):
                r_xc = l_xc + self.road_w_pix
            else:
                l_xc = r_xc - self.road_w_pix
        return l_xc, r_xc

    def porpose_win_xc(self, img, i):
        """
        提议车道检测窗位置
        获得第i个层两窗的x坐标，其来源有如下几个（按照优先级排序）：
        1. 若上一帧第 i 个窗检测到车道线，沿用x坐标
        2. 若上一帧未检出，但本帧第 i-1 个窗检出车道线，沿用其x坐标
        3. 经过上述两个步骤若只得到一侧，用一侧+/-路宽推导另一侧
        4. 用直方图计算，若只算出一侧，用一侧+/-路宽推导另一侧
        5. 用第 i-1 窗和 i-2 窗中心计算 = 2*[i-1]-[i-2]
        """
        l_xc, r_xc = None, None
        # left
        if self.lane_flag[0, i]:
            l_xc = self.lane_xc[0, i]
        elif i>1 and self.lane_flag[0, i-1]:
            l_xc = 2*self.lane_xc[0, i-1] - self.lane_xc[0, i-2]

        # right
        if self.lane_flag[1, i]:
            r_xc = self.lane_xc[1, i]
        elif i>1 and self.lane_flag[1, i-1]:
            r_xc = 2*self.lane_xc[1, i-1] - self.lane_xc[1, i-2]

        # 如果l_xc，r_xc至少一个存在，则返回
        if l_xc and r_xc:
            return l_xc, r_xc
        elif l_xc:
            r_xc = l_xc + self.road_w_pix
            return l_xc, r_xc
        elif r_xc:
            l_xc = r_xc - self.road_w_pix
            return l_xc, r_xc

        #--- 通过直方图计算
        l_xc, r_xc = self.calc_hist_xc(img, i)

        #--- 如果两个车道线都不存在
        if not l_xc:
            if i > 1:
                l_xc = 2*self.lane_xc[0, i-1] - self.lane_xc[0, i-2]
                r_xc = 2*self.lane_xc[1, i-1] - self.lane_xc[1, i-2]
            else:
                l_xc = self.lane_xc[0, i]
                r_xc = self.lane_xc[1, i]
        return int(l_xc), int(r_xc)

    def calc_hist_xc(self, img, i):
        """ 输入经过预处理的图片，以及层号，用过直方图统计输出两个检测框的 xc """
        l_xc, r_xc = None, None
        yc = self.lane_yc[i]
        yl, yh = int(yc - self.win_h//2), int(yc + self.win_h//2)
        hist_x = np.sum(img[yl:yh, :], axis=0)          # 计算 x方向直方图 [x,]
        mid_x = (self.lane_xc[0, i] + self.lane_xc[1, i]) // 2
        if mid_x > 0:
            max_i = int(np.argmax(hist_x[:mid_x]))
            if max_i>0 and max_i<self.frame_w and hist_x[max_i] > 10:
                l_xc = max_i
        if mid_x < self.frame_w:
            max_i = int(np.argmax(hist_x[mid_x:]))
            if max_i>0 and max_i<self.frame_w and hist_x[max_i] > 10:
                r_xc = max_i

        if l_xc and r_xc:
            return l_xc, r_xc
        elif l_xc:
            r_xc = l_xc + self.road_w_pix
            return l_xc, r_xc
        elif r_xc:
            l_xc = r_xc - self.road_w_pix
            return l_xc, r_xc

        return None, None

    def update_curve(self, curve_new, side):
        """ 更新拟合曲线，防止曲线突变 """
        if self.lane_curve[side] is None:
            self.lane_curve[side] = curve_new
        diff = self.lane_curve[side][1] * curve_new[1]  # y=0时的梯度 dx/dy 不能突变
        centers = np.sum(self.lane_flag[side])          # 检出点数
        if diff < -0 and centers < 10:
            return
        # ymid = - curve_new[1] / curve_new[0] / 2
        # cnt = 0
        self.lane_curve[side] = curve_new

    # def curvature(self, curve, y):
    #     """ 计算曲率半径 """
    #     return ((1 + (2*curve[0] * y * y_cmPerPixel + curve[1])**2)**1.5)\
    #                     / max(np.absolute(2*curve[0]), 0.001)


if __name__ == '__main__':
    # 透视变换
    # 快速绕圈
    # src_points = np.array([[498., 596.], [789., 596.], [250., 720.], [1050., 720.]], dtype="float32")  # 源点
    # dst_points = np.array([[300., 100.], [980., 100.], [300., 720.], [980., 720.]], dtype="float32")  # 目标点
    src_points = np.array([[399, 640], [872, 635], [261, 710], [998, 713]], dtype="float32")
    dst_points = np.array([[376, 24], [893, 24], [366, 699], [895, 702]], dtype="float32")
    # challenge_video
    # src_points = np.array([[256, 448], [475, 438], [18, 532], [791, 525]], dtype="float32")
    # dst_points = np.array([[386, 5], [841, 4], [378, 716], [860, 718]], dtype="float32")

    frameWidth = 1280  # 宽
    frameHeight = 720  # 长
    Mwarp = cv2.getPerspectiveTransform(src_points, dst_points)  # 透视变换矩阵计算
    camMat = np.array([[6.678151103217834e+02, 0, 6.430528691213178e+02],
                    [0, 7.148758960098705e+02, 3.581815819255082e+02], [0, 0, 1]])  # 相机校正矩阵
    camDistortion = np.array([[-0.056882894892153, 0.002184364631645, -0.002836821379133, 0, 0]])  # 相机失真矩阵

    # 视觉处理
    kerSz = (3, 3)  # 膨胀与腐蚀核大小
    grayThr = 125  # 二值化阈值
    roiXRatio = 0.4  # 统计x方向上histogram时选取的y轴坐标范围，以下方底边为起始点，比例定义终止位置

    # 划窗检测
    winWidth = 200  # 窗的宽度
    winNum = 20  # 窗的数目
    winThr = 8    # 单条车道线需有8个框检出车道线
    pixThr = 200  # 最小连续像素，小于该长度的被舍弃以去除噪声影响

    # 距离映射
    roadWidCm = 80      # 道路宽度 单位：cm
    roadWidPix = 660    # 透视变换后车道线像素数
    isShow = True       # 是否返回可视化图片
    cap = cv2.VideoCapture(BASE_DIR + '\\video\\test1.mp4')  # 读入视频
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)  # 设置读入图像宽
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)  # 设置读入图像长
    cap.set(cv2.CAP_PROP_FPS, 20)    # 设置读入帧率

    lane_det = laneDetect(Mwarp, camMat, camDistortion, kerSz, grayThr, frameHeight, frameWidth, roiXRatio,
                winWidth, winNum, winThr, pixThr, roadWidCm, roadWidPix, isShow)
    while True:
        ret, img = cap.read()  # 读入图片
        if ret == True:
            lane_det.spin(img)


