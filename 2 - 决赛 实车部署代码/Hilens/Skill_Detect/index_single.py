#! /usr/bin/python3.7

import cv2
import hilens
from utils import preprocess, draw_boxes, get_result
from npsocket import NumpySocket
import numpy as np


class Hilens:
    def __init__(self, checkvalue):
        # 系统初始化，your_check_value需要替换成创建技能时填写的检验值保持一致
        hilens.init(checkvalue)

        # 初始化摄像头
        self.camera = hilens.VideoCapture()
        self.display = hilens.Display(hilens.HDMI)

    def read_img(self):
        input_yuv = self.camera.read()  # 读取一帧图片(YUV NV21格式)
        input_img = cv2.cvtColor(input_yuv, cv2.COLOR_YUV2RGB_NV21)  # 转为RGB格式
        return input_img

    def show(self, img, bboxes):
        img = draw_boxes(img, bboxes)  # 画框
        output_yuv = hilens.cvt_color(img, hilens.RGB2YUV_NV21)  #转为yuv
        self.display.show(output_yuv)  # 显示到屏幕上

    def __del__(self):
        hilens.terminate()


class ObjDetect():
    def __init__(self, modelname):
        # 初始化模型，your_model_name需要替换为转换生成的om模型的名称
        model_path = hilens.get_model_dir() + modelname
        self.model = hilens.Model(model_path)

    def run(self, img):
        img_preprocess, img_w, img_h = preprocess(img)  # 缩放为模型输入尺寸
        output = self.model.infer([img_preprocess.flatten()])  # 模型推理
        bboxes = get_result(output, img_w, img_h)  # 获取检测结果
        return bboxes


def run():

    # 初始化socket
    socket_sender = NumpySocket()
    socket_sender.initialize_sender('192.168.2.1', 9999)

    # 初始化Hilens摄像头
    hl_camera = Hilens('detect')  # 根据实际情况指定

    # 初始化目标检测类
    objDet = ObjDetect('yolo3_resnet18_signal.om')

    while True:
        input_img = hl_camera.read_img()  # 读取图像
        bboxes = objDet.run(input_img)  # 目标检测
        hl_camera.show(input_img, bboxes)  # 显示目标检测框和车道线
        send_msg = socket_sender.wrap(bboxes)  # 数据打包
        socket_sender.send_array(send_msg)  # 发送数据


if __name__ == "__main__":
    run()

