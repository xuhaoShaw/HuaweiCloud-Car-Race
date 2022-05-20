import shutil
import os.path as osp
import os
#from config.yolov4_config import PROJECT_PATH, DATASET_PATH, DATA_PATH


if __name__ == '__main__':
    DATASET_PATH = './'
    DATA_PATH = '../'
    img_path = osp.join(DATASET_PATH, 'JPEGImages') + '/{:s}.jpg'
    ann_path = osp.join(DATASET_PATH, 'YOLOAnnotations') + '/{:s}.txt'
    test_file = osp.join(DATASET_PATH, 'test.txt')
    train_file = osp.join(DATASET_PATH, 'train.txt')
    out_path = osp.join(DATA_PATH, 'dataset_split')

    os.makedirs(out_path+'/images/val')
    os.makedirs(out_path+'/images/train')
    os.makedirs(out_path+'/labels/val')
    os.makedirs(out_path+'/labels/train')

    with open(test_file, 'r') as f:
        val_list = f.readlines()
    val_list = [val_name.strip() for val_name in val_list]

    with open(train_file, 'r') as f:
        train_list = f.readlines()
    train_list = [train_name.strip() for train_name in train_list]

    val_path = out_path + '/{:s}/val/{:s}.{:s}'
    for val in val_list:
        shutil.copy(img_path.format(val), val_path.format('images', val, 'jpg'))
        shutil.copy(ann_path.format(val), val_path.format('labels', val, 'txt'))
        # break

    train_path = out_path + '/{:s}/train/{:s}.{:s}'
    for tri in train_list:
        shutil.copy(img_path.format(tri), train_path.format('images', tri, 'jpg'))
        shutil.copy(ann_path.format(tri), train_path.format('labels', tri, 'txt'))
        # break