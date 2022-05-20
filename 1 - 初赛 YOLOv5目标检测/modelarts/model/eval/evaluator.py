import sys
import os
import os.path as osp
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

Base_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(Base_dir, ".."))
from eval.data_augment import Resize, RandomColorGamut
from eval.voc_eval import voc_eval, voc_eval_ap50_95
from eval.visualize import visualize_boxes
from eval.tools import non_max_suppression
import yolov5_config as cfg
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
current_milli_time = lambda: int(round(time.time() * 1000))


class Evaluator(object):
    def __init__(self, model=None, conf_thresh=None, nms_thresh=None, ap50_95=True):
        self.classes        =   cfg.Customer_DATA["CLASSES"]    # default: use customer dataset
        if conf_thresh is None:
            self.conf_thresh =  cfg.CONF_THRESH
        else:
            self.conf_thresh = conf_thresh
        if nms_thresh is None:
            self.nms_thresh     =   cfg.NMS_THRESH       # 0.3
        else:
            self.nms_thresh = nms_thresh

        self.test_size      =   cfg.IMG_SIZE
        self.model          =   model
        self.device         =   next(model.parameters()).device
        self.class_record   =   defaultdict(list)               # detected objs categorized by classes

        self.max_visual_img =   20 # max num of pred images to be visualized
        self.cnt_visual_img =   0  # num of pred images be visualized
        self.inference_time =   0.0

        self.pred_image_path    = osp.join(cfg.EVAL_DATASET_PATH, "pred_images")
        self.ap_calculator = voc_eval_ap50_95 if ap50_95 else voc_eval

    def calc_APs(self):
        if not osp.exists(self.pred_image_path):
            os.mkdir(self.pred_image_path)
        # read images from test.txt
        img_list_file = osp.join( cfg.EVAL_DATASET_PATH,  "test.txt" )
        with open(img_list_file, "r") as f:
            img_list = [img.strip() for img in f.readlines()]
        # predict all imgs, save result in class_record
        pool = ThreadPool(multiprocessing.cpu_count())
        with tqdm(total=len(img_list), ncols=120, smoothing=0.9 ) as tq:
            for i, _ in enumerate(pool.imap_unordered(self.predict_and_save, img_list)):
                tq.update()
        self.inference_time = 1.0 * self.inference_time / len(img_list)
        # calc mAP
        APs = {}
        for cls, data in self.class_record.items():
            img_idxs = np.array([rec[0] for rec in data])
            bbox_conf = np.array([rec[1] for rec in data], dtype=np.float32)
            AP = self.ap_calculator((img_idxs, bbox_conf), cls_name=cls)
            APs[cls] = AP
        return APs, self.inference_time

    def predict_and_save(self, img_path):
        # img = RandomColorGamut()(cv2.imread(img_path))
        img = cv2.imread(img_path)
        # predict all valid bboxes in img [N, 6]
        bboxes_prd = self.predict(img)
        # visualization
        img_idx = Path(img_path).stem
        if bboxes_prd.shape[0] != 0  and self.cnt_visual_img < self.max_visual_img:
            self.visualize(img, bboxes_prd, img_idx, save=True)
            self.cnt_visual_img += 1

        # save to class_record
        for bbox in bboxes_prd:
            cls_name = self.classes[int(bbox[5])]
            self.class_record[cls_name].append([img_idx, bbox[:5]])

    def predict(self, img):
        org_shape = img.shape[:2]
        img = self.__transform(img)  # [1, C, test_size, test_size]
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            pred_decode, _ = self.model(img)   # p_d
            self.inference_time += current_milli_time() - start_time
        # p_d: [Sigma0~2[batchsize x grid[i] x grid[i] x anchors(3)], x+y+w+h+conf+cls_6(11)]
        # [N, 6(xmin, ymin, xmax, ymax, score, class)]
        pred = non_max_suppression(pred_decode, self.conf_thresh, self.nms_thresh, multi_label=True)
        pred = pred[0].cpu().numpy() # xyxy conf cls
        bboxes = self.resize_pb(pred, self.test_size, org_shape)
        return bboxes

    def resize_pb(
        self, pred_bbox, test_input_size, org_img_shape
    ):
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

        return pred_bbox

    def __transform(self, img):
        img = Resize((self.test_size, self.test_size), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float().to(self.device)

    def visualize(self, img, pred, img_name=None, save=False, score_thresh=0.45):
        if len(pred) > 0:
            bboxes  = pred[:, :4]
            cls_ids = pred[:, 5].round().astype(np.int32)
            scores  = pred[:, 4]
            visualize_boxes(
                image=img, boxes=bboxes, labels=cls_ids, probs=scores,
                class_labels=self.classes, min_score_thresh=score_thresh)
        if save:
            save_path = osp.join(self.pred_image_path, "{}.jpg".format(img_name))
            cv2.imwrite(save_path, img)
        return img


if __name__ == '__main__':
    import time
    from models.yolo import Model

    # load model
    print("loading weight file from : {}".format(cfg.EVAL_WEIGHT_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cfg.EVAL_WEIGHT_PATH, map_location=device)
    print(ckpt['model'].yaml)
    yolo = Model(ckpt['model'].yaml, ch=3, nc=6).to(device)
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    yolo.load_state_dict(state_dict, strict=False)  # load
    print('Transferred %g/%g items from %s' % (len(state_dict), len(yolo.state_dict()), cfg.EVAL_WEIGHT_PATH))

    # eval
    evalutaor = Evaluator(yolo, ap50_95=True)

    start = time.time()
    mAP = np.zeros(2)
    Aps, inference_time = evalutaor.calc_APs()
    for k, v in Aps.items():
        print('{:<20s}: ap50 {:.3f} | ap50_95 {:.3f}'.format(k, v[0], v[1]))
        mAP += v
    print('mAP', ' '*15, ': ap50 {0[0]:.3f} | ap50_95 {0[1]:.3f}'.format(mAP/6.0))
    print('inference time: {}ms' .format(inference_time))
    print('total time    : %d' % (time.time() - start))

