from pathlib import Path
import yaml
DATA_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = DATA_DIR.joinpath('dataset.yaml')    # 数据集配置文件
with DATA_YAML.open('r') as f:
    data_cfg = yaml.safe_load(f)

# 数据集配置
DATASET_PATH = DATA_DIR.parent.joinpath(data_cfg['path'])   # 数据集路径
IMG_PATH = DATASET_PATH / 'JPEGImages'  # 图片路径
VOC_PATH = DATASET_PATH / 'Annotations' # voc 格式标签路径
COCO_PATH = DATASET_PATH / 'COCOAnnos'  # coco 格式标签路径
YOLO_PATH = DATASET_PATH / 'YOLOAnnos'  # yolo 格式标签路径
CLASS_PATH = DATASET_PATH / 'ClassAnnos'  # 按类别分类标签路径
split_percent = data_cfg['split_percent'] # 数据集划分比例

Customer_DATA = {
    "NUM": data_cfg['nc'],  # your dataset number
    "CLASSES": data_cfg['names']  # your dataset class
}