"""
convert voc format annotations to coco format annotations, and save to dataset/COCOAnnos/
"""
import os
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from data_config import Customer_DATA, COCO_PATH, DATASET_PATH, VOC_PATH
from tqdm import tqdm

START_BOUNDING_BOX_ID = 1
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {cls: i for i, cls in enumerate(Customer_DATA['CLASSES'])}

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in tqdm(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
            filename_o = Path(xml_file).stem + '.jpg'
            assert filename_o != filename, "filename does not match"
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text.split('.')[0]) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text.split('.')[0]) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text.split('.')[0])
            ymax = int(get_and_check(bndbox, "ymax", 1).text.split('.')[0])
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":
    import os
    train_file = DATASET_PATH / 'train.txt'
    test_file = DATASET_PATH / 'test.txt'
    assert train_file.exists() and test_file.exists(), "Please run gen_img_index_file.py first."

    # train
    train_json = str( COCO_PATH / 'train.json' )
    train_xmls = []
    with train_file.open('r') as f:
        for idx in f.readlines():
            idx = Path(idx.strip()).stem
            train_xmls.append(str(VOC_PATH) + os.sep + idx + '.xml')
    convert(train_xmls, train_json)
    print("Success: {}".format(train_json))

    # test
    test_json = str( COCO_PATH / 'test.json' )
    test_xmls = []
    with test_file.open('r') as f:
        for idx in f.readlines():
            idx = Path(idx.strip()).stem
            test_xmls.append(str(VOC_PATH) + os.sep + idx + '.xml')
    convert(test_xmls, test_json)
    print("Success: {}".format(test_json))
