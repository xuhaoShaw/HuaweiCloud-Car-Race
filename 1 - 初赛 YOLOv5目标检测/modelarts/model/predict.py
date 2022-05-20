import os
import os.path as osp
import torch
import numpy as np
from time import time
current_milli_time = lambda: int(round(time() * 1000))
from models.yolo import Model
import cv2
from eval.evaluator import Evaluator
from yolov5_config import EVAL_DATASET_PATH, EVAL_WEIGHT_PATH


if __name__ == '__main__':
    # image list
    img_path = EVAL_DATASET_PATH +'\\images\\{:s}.jpg'
    img_list = np.array([x.split('.')[0] for x in os.listdir(osp.join(EVAL_DATASET_PATH, 'images'))])
    np.random.shuffle(img_list)

    # load model
    print("loading weight file from : {}".format(EVAL_WEIGHT_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(EVAL_WEIGHT_PATH, map_location=device)
    yolo = Model(ckpt['model'].yaml, ch=3, nc=6).to(device)
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    yolo.load_state_dict(state_dict, strict=False)  # load
    print('Transferred %g/%g items from %s' % (len(state_dict), len(yolo.state_dict()), EVAL_WEIGHT_PATH))


    # predict
    predictor = Evaluator(yolo, conf_thresh=0.1, nms_thresh=0.3)

    cv2.namedWindow('Predict', flags=cv2.WINDOW_NORMAL)
    for img_name in img_list:
        # read image
        img = cv2.imread(img_path.format(img_name))
        print('Show: ' + img_path.format(img_name))
        print('Continue? ([y]/n)? ')
        pred = predictor.predict(img)
        print(pred)
        img = predictor.visualize(img, pred, score_thresh=0.)
        cv2.imshow('Predict', img)
        c = cv2.waitKey()
        if c in [ord('n'), ord('N')]:
            exit()


