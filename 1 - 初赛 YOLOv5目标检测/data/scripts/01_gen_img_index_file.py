"""
Generate train.txt and test.txt in DATASET_PATH
"""
import numpy as np
from tqdm import tqdm
from data_config import DATASET_PATH, IMG_PATH, split_percent

if __name__ == "__main__":
    assert IMG_PATH.exists(), f"Path {IMG_PATH} does not exists"
    train_index_file = DATASET_PATH / 'train.txt'
    test_index_file =  DATASET_PATH / 'test.txt'
    # 获取所有图片路径
    img_list = [str(img.resolve()) for img in IMG_PATH.glob('*.jpg')]
    # shuffle and split
    index = np.random.permutation(len(img_list))
    split_index = int(len(index) * split_percent)

    train_list  = np.array(img_list)[ index[:split_index] ]
    test_list   = np.array(img_list)[ index[split_index:] ]

    with open(train_index_file, "w") as f:
        for image_id in tqdm(train_list):
            f.write(image_id + "\n")

    with open(test_index_file, "w") as f:
        for image_id in tqdm(test_list):
            f.write(image_id + "\n")
