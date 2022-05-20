try:
    from model_service.pytorch_model_service import PTServingBaseService
except:
    from tools import PTServingBaseService

import torch.nn as nn
import torch
import json
import numpy as np
import torchvision.transforms as transforms
import cv2
from models.yolo import Model
import yolov5_config as cfg
from tools import Resize, non_max_suppression

import os.path as osp

class Yolov5Service(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(Yolov5Service, self).__init__(model_name, model_path)
        self.base_dir =  osp.dirname(osp.realpath(__file__))
        # load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(cfg.WEIGHT_PATH, map_location=self.device)
        self.yolo = Model(ckpt['model'].yaml, ch=3, nc=6).to(self.device)
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        self.yolo.load_state_dict(state_dict, strict=False)  # load
        self.yolo.eval()
        print('Transferred %g/%g items from %s' % (len(state_dict), len(self.yolo.state_dict()), cfg.WEIGHT_PATH))

        self.test_size = cfg.IMG_SIZE
        self.org_shape = (0, 0)
        self.conf_thresh = cfg.CONF_THRESH
        self.nms_thresh = cfg.NMS_THRESH
        self.Classes = np.array(cfg.Customer_DATA["CLASSES"])

    def _preprocess(self, data):
        pro_data = {}
        image_dict = data['images']
        input_batch = []
        for img_name, img_content in image_dict.items():
            img = self.__imread(img_content)
            self.org_shape = img.shape[:2]
            img = self.__transform(img)
            input_batch.append(img)
        pro_data['images'] = torch.stack(input_batch, dim=0)

        return pro_data

    def _inference(self, data):
        result = {}
        with torch.no_grad():
            input_batch = data['images']
            # v - [1, C, test_size, test_size]
            pred, _ = self.yolo(input_batch)
            # pred - [N, 6(xmin, ymin, xmax, ymax, score, class)]
            pred = non_max_suppression(pred, self.conf_thresh, self.nms_thresh, multi_label=True)
            pred = pred[0].cpu().numpy() # xyxy conf cls
            pred = self.resize_pb( pred, self.test_size, self.org_shape)
            result['images'] = pred
        return result

    def _postprocess(self, data):
        result = {}
        for k, v in data.items():
            cls_idx = v[:, 5].astype(np.int32)
            detection_classes = self.Classes[cls_idx]
            detection_boxes = v[:, [1,0,3,2]].round().astype(np.int32)
            detection_scores = v[:, 4]
            result = {
                'detection_classes': detection_classes.tolist(),
                'detection_boxes'  : detection_boxes.tolist(),
                'detection_scores' : detection_scores.tolist()
            }
        return result

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


    def __transform(self, img):
        img = Resize((self.test_size, self.test_size), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img).float().to(self.device)
        # return torch.from_numpy(img[np.newaxis, ...]).float().to(self.device)

    def __imread(self, img_byteio):
        return cv2.imdecode(
            np.frombuffer(img_byteio.getbuffer(), np.uint8),
            cv2.IMREAD_COLOR
        )


if __name__ == "__main__":
    import numpy as np
    from io import BytesIO
    from yolov5_config import MODEL_PATH
    img_url = MODEL_PATH + '\\test.jpg'

    with open(img_url, 'rb') as f:
        a = BytesIO(f.read())

    images = {'img1': a}
    data = {
        'images':images
    }

    server = Yolov5Service('', '')
    data = server._preprocess(data)
    result = server._inference(data)
    result = server._postprocess(result)
    print(result['detection_boxes'])


