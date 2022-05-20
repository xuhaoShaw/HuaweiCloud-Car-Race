import sys

import albumentations as alb
import cv2
import os
import numpy as np
import pathlib

dataPath = r"D:\Program\HuaweiComp\data\Datav5"  # 数据位置
savePath = r"D:\Program\HuaweiComp\data\Datav6"  # 存储位置

# 选用的数据增强方法
# GaussianNoise: <All class>
# MotionBlur：<All class>
# Affine: <All class>
# RandomBrightness: <All class>
transform = alb.Compose([alb.Affine(scale=1.0, translate_percent=(-0.3, 0.3), translate_px=None, rotate=(-20.0, 20.0),
                                    shear=(-5.0, 5.0), interpolation=cv2.INTER_LINEAR, cval=0, cval_mask=0,
                                    mode=cv2.BORDER_CONSTANT, fit_output=False, p=0.8),
                         alb.RandomBrightness(limit=0.3, p=0.6),
                         alb.GaussNoise(var_limit=(0.0, 50.0), mean=0, per_channel=False, p=0.5),
                         alb.MotionBlur(blur_limit=(3, 7), p=0.5)])
max_iteration = 5
count = 1

if __name__ == '__main__':
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            filename = os.path.join(root, file)
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for iteration in range(max_iteration):
                img_transformed = transform(image=img)['image']
                savename = os.path.join(savePath, str(count) + '.jpg')
                print(savename)
                cv2.imwrite(savename, img_transformed, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                count = count + 1




