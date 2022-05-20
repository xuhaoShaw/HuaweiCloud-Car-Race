import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent

import cv2
import numpy as np
import mindspore
from mindspore import load_checkpoint, load_param_into_net, context, Tensor, dtype, ops
from service_utils.tools import Resize,  batch_nms
from service_utils.general import colorprint
from service_utils.visualize import visualize_boxes

import yolov5_config as cfg
from models.yolov5s import Model
#
from tqdm import tqdm
from service_utils.voc_eval import voc_eval_ap50_95
from collections import defaultdict
import time
current_milli_time = lambda: int(round(time.time() * 1000))

class Evaluator():

    def __init__(self):
        super(Evaluator, self).__init__()
        # 加载模型
        param_dict = load_checkpoint(str(cfg.WEIGHT_PATH))
        self.test_size = 1024
        print(f'Image size: {self.test_size}')
        self.yolo = Model(bs=1, img_size=self.test_size)  # batch size 默认为 1
        not_load_params = load_param_into_net(self.yolo, param_dict)

        self.org_shape = (0, 0)
        self.conf_thresh = cfg.CONF_THRESH
        self.nms_thresh = cfg.NMS_THRESH
        self.Classes = np.array(cfg.Customer_DATA["CLASSES"])
        self.resize = Resize((self.test_size, self.test_size), correct_box=False)
        self.inference_time = 0

    def _preprocess(self, img):
        self.org_shape = img.shape[:2]
        img = self.resize(img).transpose(2, 0, 1) # c,w,h
        return Tensor(img[np.newaxis, ...], dtype=dtype.float32) # bs,c,w,h

    def _inference(self, data):
        # infer
        start_time = current_milli_time()  ### TODO
        preds = self.yolo(data)  # list of Tensor
        self.inference_time += current_milli_time() - start_time  ###
        pred=preds[6][0:1,...].asnumpy()  # Tensor [1, N, 11(xywh)]
        # nms
        pred = batch_nms(pred, self.conf_thresh, self.nms_thresh)  # numpy [1, n, 6(xyxy)]
        pred = pred[0]
        # resize to original size
        bbox = self.resize_pb( pred, self.test_size, self.org_shape)
        return bbox

    def predict(self, data):
        return self._inference(self._preprocess(data))

    def resize_pb(self, pred_bbox, test_input_size, org_img_shape):
        """
        input: pred_bbox - [n, x+y+x+y+conf+cls(6)]
        Resize to origin size
        output: pred_bbox [n1, x+y+x+y+conf+cls (6)]
        """
        pred_coor = pred_bbox[:, :4] #  xyxy
        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0*test_input_size/org_w, 1.0*test_input_size/org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)Crop off the portion of the predicted Bbox that is beyond the original image
        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :2], [0, 0]),
                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
            ],
            axis=-1,
        )

        return np.concatenate([pred_coor, pred_bbox[:, 4:]], axis=-1)


    def __imread(self, img_byteio):
        """
        read image from byteio
        """
        return cv2.imdecode(
            np.frombuffer(img_byteio.getbuffer(), np.uint8),
            cv2.IMREAD_COLOR
        )

    def visualize(self, img, pred, img_name=None, save=False, score_thresh=0, save_path=None):
        if len(pred) > 0:
            bboxes  = pred[:, :4]
            cls_ids = pred[:, 5].round().astype(np.int32)
            scores  = pred[:, 4]
            visualize_boxes(
                image=img, boxes=bboxes, labels=cls_ids, probs=scores,
                class_labels=self.Classes, min_score_thresh=score_thresh)
        if save:
            cv2.imwrite(save_path, img)
        return img

    def calc_APs(self):
        # read images from test.txt
        self.class_record   =   defaultdict(list)
        img_list_file = cfg.DATASET_PATH / "test.txt"
        with open(img_list_file, "r") as f:
            lines = f.readlines()
            img_list = [line.strip() for line in lines]
        # predict all imgs, save result in class_record
        for img_path in tqdm(img_list, ncols=120, smoothing=0.9 ):
            self.predict_and_save(img_path)
        self.inference_time = 1.0 * self.inference_time / len(img_list)
        # calc mAP
        APs = {}
        for cls, data in self.class_record.items():
            img_idxs = np.array([rec[0] for rec in data])
            bbox_conf = np.array([rec[1] for rec in data], dtype=np.float32)
            AP = voc_eval_ap50_95((img_idxs, bbox_conf), cls_name=cls)
            APs[cls] = AP
        mAP = np.mean(list(APs.values()), axis=0)  # [mAP@0.5, mAP@0.5:0.95]
        return APs, mAP, self.inference_time

    def predict_and_save(self, img_path):
        # find img file
        img = cv2.imread(img_path)
        img_idx = Path(img_path).stem
        # predict all valid bboxes in img [N, 6]
        bboxes_prd = self.predict(img)
        # save to class_record
        for bbox in bboxes_prd:
            cls_name = self.Classes[int(bbox[5])]
            self.class_record[cls_name].append([img_idx, bbox[:5]])



if __name__ == "__main__":
    # image list
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # predict
    predictor = Evaluator()
    APs, mAP, infer_time = predictor.calc_APs()
    for k, v in APs.items():
        print('{:<20s}: ap50 {:.3f} | ap50_95 {:.3f}'.format(k, v[0], v[1]))
    print('mAP', ' '*15, ': ap50 {0[0]:.3f} | ap50_95 {0[1]:.3f}'.format(mAP))
    print('inference time: {}ms' .format(infer_time))



