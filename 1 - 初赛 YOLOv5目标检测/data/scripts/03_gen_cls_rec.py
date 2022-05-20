"""
split test annotations according to class name
and save at data/ClassRecords/[classname].txt
each line: 'img_idx x1,y1,x2,y2,difficult'
"""
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
from data_config import VOC_PATH, CLASS_PATH, DATASET_PATH


def parse_anno(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text.split('.')[0]),
            int(bbox.find("ymin").text.split('.')[0]),
            int(bbox.find("xmax").text.split('.')[0]),
            int(bbox.find("ymax").text.split('.')[0]),
        ]
        objects.append(obj_struct)
    return objects


if __name__ == "__main__":
    # 类记录路径
    CLASS_PATH.mkdir(exist_ok=True)

    with (DATASET_PATH/'test.txt').open('r') as f:
        img_list = f.readlines()

    Cls_Record = defaultdict(list)
    for img in img_list:
        img_name = Path(img).stem
        objs = parse_anno(str(VOC_PATH.joinpath(img_name+'.xml')))
        line = img_name + ' {:d},{:d},{:d},{:d},{:d}\n'
        for obj in objs:
            cls_name = obj["name"]
            difficult = obj["difficult"]
            bbox = obj["bbox"]
            Cls_Record[cls_name].append(line.format(*bbox, difficult))

    for cls_name, lines in Cls_Record.items():
        print("class: {:<20s}, total num: {:d}".format(cls_name, len(lines)))
        with CLASS_PATH.joinpath(cls_name+'.txt').open('w') as f:
            f.write(''.join(lines))





