try:
    from model_service.model_service import SingleNodeService
except:
    from service_utils.tools import SingleNodeService
from pathlib import Path
BASE_DIR = Path(__file__).parent

import cv2
import numpy as np
import mindspore
from mindspore import load_checkpoint, load_param_into_net, context, Tensor, dtype, ops
from service_utils.tools import Resize,  batch_nms

import yolov5_config as cfg
from models.yolov5s import Model

import threading
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

print(f"Model Path: {str(cfg.MODEL_PATH)}")

class Yolov5Service(SingleNodeService):

    def __init__(self, model_name, model_path):
        super(Yolov5Service, self).__init__(model_name, model_path)
        # 加载模型
        param_dict = load_checkpoint(str(cfg.WEIGHT_PATH))
        self.test_size = 1024
        print(f'********Image size: {self.test_size}')
        self.yolo = Model(bs=1, img_size=self.test_size)  # batch size 默认为 1
        not_load_params = load_param_into_net(self.yolo, param_dict)
        self.yolo.set_train(False)

        self.org_shape = (0, 0)
        self.conf_thresh = cfg.CONF_THRESH
        self.nms_thresh = cfg.NMS_THRESH
        self.Classes = np.array(cfg.Customer_DATA["CLASSES"])
        self.resize = Resize((self.test_size, self.test_size), correct_box=False)
        self.network_warmup()

    def network_warmup(self):
        # 模型预热，否则首次推理的时间会很长
        logger.info("warmup network ... \n")
        images = np.array(np.random.randn(1, 3, 1024, 1024), dtype=np.float32)
        inputs = Tensor(images, mindspore.float32)
        inference_result = self.yolo(inputs)
        logger.info("warmup network successfully ! \n")

    def _preprocess(self, data):
        pro_data = {}
        image_dict = data['images']
        input_batch = []
        for img_name, img_content in image_dict.items():
            img = self.__imread(img_content)
            self.org_shape = img.shape[:2]
            img = Tensor(   self.resize(img).transpose(2, 0, 1),
                            dtype=dtype.float32 )
            input_batch.append(img)
        pro_data['images'] = ops.Stack()(input_batch)

        return pro_data


    def _inference(self, data):
        result = {}
        input_batch = data['images']
        # infer
        preds = self.yolo(input_batch)  # list of Tensor
        pred = preds[6][0:1,...].asnumpy()  # Tensor [1, N, 11(xywh)]
        # print(pred[0, :10])
        # nms
        pred = batch_nms(pred, self.conf_thresh, self.nms_thresh)  # numpy [1, n, 6(xyxy)]
        pred = pred[0]
        # resize to original size
        bbox = self.resize_pb( pred, self.test_size, self.org_shape)
        result['images'] = bbox
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


    def __imread(self, img_byteio):
        """
        read image from byteio
        """
        return cv2.imdecode(
            np.frombuffer(img_byteio.getbuffer(), np.uint8),
            cv2.IMREAD_COLOR
        )


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    import numpy as np
    from io import BytesIO
    from yolov5_config import DATASET_PATH
    img_url = DATASET_PATH  / 'JPEGImages/2.jpg'

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
