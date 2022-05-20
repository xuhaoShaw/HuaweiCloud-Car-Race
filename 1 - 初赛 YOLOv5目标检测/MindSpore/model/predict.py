import sys
from pathlib import Path
import cv2
import numpy as np
from mindspore import context

import yolov5_config as cfg
from evaluate import Evaluator


if __name__ == "__main__":
    # image list
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    IMG_PATH = cfg.DATASET_PATH.joinpath('JPEGImages')
    img_list = np.array([x.name for x in IMG_PATH.iterdir()])
    np.random.shuffle(img_list)
    img_list[0] = '836.jpg'
    # predict
    predictor = Evaluator()

    cv2.namedWindow('Predict', flags=cv2.WINDOW_NORMAL)
    for img_name in img_list:
        # read image
        img = cv2.imread(str(IMG_PATH/img_name))
        print('Show: ' + str(IMG_PATH/img_name))
        print('Continue? ([y]/n)? ')
        pred = predictor.predict(img)
        print(pred)
        img = predictor.visualize(img, pred, score_thresh=0.)
        cv2.imshow('Predict', img)
        c = cv2.waitKey()
        if c in [ord('n'), ord('N')]:
            exit()


