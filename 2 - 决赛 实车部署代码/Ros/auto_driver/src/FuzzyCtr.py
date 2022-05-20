#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class FuzzyCtr():
    def __init__(self, input1, input2, rules):
        """
        初始化
        :param input1: 控制变量1的n个取值如：[-5, -2, -1, 0, 1, 2, 5]
        :param input2: 控制变量2的n个取值
        :param rules: 规则矩阵,横向代表x(axis=1),纵向代表y(axis=0)
        """
        # 后面补充一个大数
        self.x_range = np.concatenate([np.array(input1), [input1[-1]+10000]]).astype(float)
        self.y_range = np.concatenate([np.array(input2), [input2[-1]+10000]]).astype(float)
        self.rules = np.array(rules).astype(float)

    def control(self, x, y):
        """ 计算控制变量 """
        # 限制幅值
        x = self.clump(x, self.x_range[0], self.x_range[-2])
        y = self.clump(y, self.y_range[0], self.y_range[-2])
        # 查找输入在rules矩阵里的坐标
        xi = self.search_idx(x, self.x_range)
        yi = self.search_idx(y, self.y_range)
        # 计算隶属度
        mu_x = np.array([self.x_range[xi+1]-x, x-self.x_range[xi]]) / [self.x_range[xi+1]-self.x_range[xi]]
        mu_y = np.array([self.y_range[yi+1]-y, y-self.y_range[yi]]) / [self.y_range[yi+1]-self.y_range[yi]]
        # 计算隶属度矩阵
        mu_matrix = np.matmul(mu_y[:, None], mu_x[None, :])
        # 计算输出
        rules_act = self.rules[yi:yi+2, xi:xi+2]
        output = np.sum(mu_matrix*rules_act)
        return output


    def search_idx(self, x, x_range):
        """ 若 x 在 x_range 中的第i个左开右闭区间[x_range[i], x_range[i+1])，返回i """
        for i in range(len(x_range)-1):
            if x >= x_range[i] and x < x_range[i+1]:
                return i


    def clump(self, x, low, high):
        """ 限制幅值 """
        if x < low:
            return low
        if x > high:
            return high
        return x

class FuzzyCtr1D():
    def __init__(self, input1, rules):
        """
        初始化
        :param input1: 控制变量1的n个取值如：[-5, -2, -1, 0, 1, 2, 5]
        :param rules: 规则矩阵,横向代表x(axis=1),纵向代表y(axis=0)
        """
        # 后面补充一个大数
        self.x_range = np.concatenate([np.array(input1), [input1[-1]+10000]]).astype(float)
        self.rules = np.array(rules).astype(float)

    def control(self, x):
        """ 计算控制变量 """
        # 限制幅值
        x = self.clump(x, self.x_range[0], self.x_range[-2])
        # 查找输入在rules矩阵里的坐标
        xi = self.search_idx(x, self.x_range)
        # 计算隶属度
        mu_x = np.array([self.x_range[xi+1]-x, x-self.x_range[xi]]) / [self.x_range[xi+1]-self.x_range[xi]]
        # 计算输出
        rules_act = self.rules[xi:xi+2]
        output = np.sum(mu_x*rules_act)
        return output


    def search_idx(self, x, x_range):
        """ 若 x 在 x_range 中的第i个左开右闭区间[x_range[i], x_range[i+1])，返回i """
        for i in range(len(x_range)-1):
            if x >= x_range[i] and x < x_range[i+1]:
                return i


    def clump(self, x, low, high):
        """ 限制幅值 """
        if x < low:
            return low
        if x > high:
            return high
        return x


# # 定义车道线循迹控制器
# bias_range = [-50, -30, -15, 0, 15, 30, 50]
# slope_range = [-3, -2, -1, 0, 1, 2, 3]
#         # -50, -30, -15,   0,  15,  30, 50
# rules = [
#          np.array([-30, -25, -15, 0, 15, 25, 30])-20,  # -3
#          np.array([-30, -25, -15, 0, 15, 25, 30])-10,  # -2
#          np.array([-30, -25, -15, 0, 15, 25, 30])-5,  # -1
#          np.array([-30, -25, -15, 0, 15, 25, 30]),  # -0
#          np.array([-30, -25, -15, 0, 15, 25, 30])+5,  # 1
#          np.array([-30, -25, -15, 0, 15, 25, 30])+10,  # 2
#          np.array([-30, -25, -15, 0, 15, 25, 30])+20,  # 3
#         ]

# FollowLineCtr = FuzzyCtr(bias_range, slope_range, rules)


if __name__ == '__main__':
    bias_range = [-50, -30, -15, 0, 15, 30, 50]
    gear_range = [0]
            # -50, -30, -15,   0,  15,  30, 50
    rules = np.array([-30, -25, -15, 0, 15, 25, 30])


    FollowLineCtr = FuzzyCtr(bias_range, gear_range, rules)
    Followctr =  FuzzyCtr1D(bias_range, rules)
    y = 0
    for x in range(-30, 30, 5):
        print(x, Followctr.control(x))