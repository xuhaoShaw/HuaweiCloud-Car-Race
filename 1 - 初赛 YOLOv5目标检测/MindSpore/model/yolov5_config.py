# coding=utf-8
import os.path as osp
from pathlib import Path

# 服务配置
WEIGHT = 'yolov5s_datav3_1024.ckpt'               # 模型权重
MODEL_PATH = Path(__file__).resolve().parent      # mindspore路径
WEIGHT_PATH = MODEL_PATH / 'weights' / WEIGHT

# try:
#     IMG_SIZE = int(WEIGHT.split('.')[0].split('_')[-1])       # 图片尺寸
# except:
#     IMG_SIZE = 640

# 本地测试配置
DATASET_PATH = MODEL_PATH / '../../data/datav3'  # 测试用的数据集

# 推理配置
NMS_THRESH =0.1
CONF_THRESH = 0.1

Customer_DATA = {
    "NUM": 6,  # your dataset number
    "CLASSES": [
        "red_stop",
        "green_go",
        "yellow_back",
        "speed_limited",
        "speed_unlimited",
        "pedestrian_crossing",
        ],  # your dataset class
}
