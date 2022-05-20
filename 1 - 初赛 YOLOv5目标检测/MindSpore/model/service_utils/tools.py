import numpy as np
import time
import cv2
from mindspore import Tensor, ops


class PTServingBaseService():
    """ for local test """
    def __init__(self, arg1, arg2) -> None:
        pass

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def batch_nms(preds, conf_thres=0.25, iou_thres=0.45, agnostic=False,
                         max_det=300):
    """ Runs Non-Maximum Suppression (NMS) on batchs
    Params:
        - preds: [batchsize, N, 11(xywh, conf, cls)]
    Returns:
        - output: [batchsize, n, 6]
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = preds.shape[0]  # batch size
    nc = preds.shape[2] - 5  # number of classes
    xc = preds[..., 4] > conf_thres  # candidates: batch x N x 1

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 2048  # (pixels) minimum and maximum box width and height
    max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    output = [np.zeros((0, 6))] * bs  # batch x 0 x 6
    for xi, x in enumerate(preds):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence: n x 11

        # If none remain process, next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf = x[:, 5:].max(axis=1, keepdims=True)
        cls = x[:, 5:].argmax(axis=1)
        x = np.concatenate((box, conf, cls[:, None]), axis=1)[conf.reshape((-1)) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        cls_bias = x[:, 5:6] * (0 if agnostic else max_wh)  # classes bias
        boxes, scores = x[:, :4] + cls_bias, x[:, 4].copy()  # boxes (offset by class), scores
        idx = nms(boxes, scores, iou_thres)  # NMS
        if idx.shape[0] > max_det:  # limit detections
            idx = idx[:max_det]

        output[xi] = x[idx]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output


def nms(bboxes, scores, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes:
        bboxes的shape为(N, 4)，存储格式为(xmin, ymin, xmax, ymax)
        其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的
    :return: index
    """
    index = []
    mask = np.full((bboxes.shape[0],), True, dtype=np.bool8)  # remain bboxes
    while mask.any():
        max_idx = np.argmax(scores)
        index.append(max_idx)
        scores[max_idx] = 0
        mask[max_idx] = False

        iou = Giou_xyxy_numpy(bboxes[max_idx:max_idx+1, :], bboxes[mask])
        assert method in ['nms', 'soft-nms']
        weight = np.ones((len(iou),), dtype=np.float32)
        if method == 'nms':
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0
        if method == 'soft-nms':
            weight = np.exp(-(1.0 * iou ** 2 / sigma))
        scores[mask] = scores[mask] * weight
        mask = scores > 0

    return np.array(index)


def Giou_xyxy_numpy(boxes1, boxes2):
    '''
    cal GIOU of two boxes or batch boxes
    such as: (1)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[15,15,25,25]])
            boxes2 = np.asarray([[5,5,10,10]])
            and res is [-0.49999988  0.25       -0.68749988]
            (2)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            and res is [1. 1. 1.]
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # cal GIOU
    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
    return GIOU


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes=None):
        h_org, w_org, _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(
            1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org
        )
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh : resize_h + dh, dw : resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class SingleNodeService():
    """ for local test """
    def __init__(self, arg1, arg2) -> None:
        pass
