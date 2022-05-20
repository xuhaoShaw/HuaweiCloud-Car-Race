"""
convert voc format annotations to yolo format annotations, and save to dataset/YOLOAnnos/
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

from data_config import VOC_PATH, YOLO_PATH, DATASET_PATH
from data_config import Customer_DATA

def parse_anno(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    s = tree.find("size")
    w = float(s.find("width").text)
    h = float(s.find("height").text)
    lines = []
    line_pattern = '{:d} {:f} {:f} {:f} {:f}\n'
    for obj in tree.findall("object"):
        cls = Customer_DATA['CLASSES'].index(obj.find("name").text)
        bbox = obj.find("bndbox")
        xyxy = np.array([
            float(bbox.find("xmin").text)/w,
            float(bbox.find("ymin").text)/h,
            float(bbox.find("xmax").text)/w,
            float(bbox.find("ymax").text)/h
        ])
        xywh = xyxy2xywh(xyxy)
        lines.append(line_pattern.format(cls, *list(xywh)))
    return lines

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h] and normalize to 1
    y = np.zeros_like(x)
    y[0] = (x[0] + x[2]) / 2.0
    y[1] = (x[1] + x[3]) / 2.0
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y


if __name__ == '__main__':

    path = [DATASET_PATH / 'test.txt', DATASET_PATH / 'train.txt']

    YOLO_PATH.mkdir(exist_ok=True)

    xml_list = []
    for p in path:
        with p.open('r') as f:
            xmls = [ VOC_PATH / Path(file).with_suffix('.xml').name for file in f.readlines()]
            xml_list += xmls

    for xml in tqdm(xml_list):
        lines = parse_anno(str(xml.resolve()))
        label = YOLO_PATH / xml.with_suffix('.txt').name
        with label.open('w') as f :
            f.writelines(lines)