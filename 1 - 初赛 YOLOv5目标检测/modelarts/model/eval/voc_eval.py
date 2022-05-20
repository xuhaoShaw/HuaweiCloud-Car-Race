# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

from collections import defaultdict
import xml.etree.ElementTree as ET
import sys
import os.path as osp
Base_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(Base_dir, ".."))
from yolov5_config import EVAL_DATASET_PATH as DATASET_PATH
import numpy as np


def read_cls_anno(cls_name):
    """
    read all annotations of a class to calculate AP
    return a dict, {img_idx: array[[x1,y1,x2,y2,difficult], ...]}
    """
    cls_record = defaultdict(list)
    cls_path = osp.join(DATASET_PATH, 'ClassAnnos', cls_name+'.txt')
    with open(cls_path, "r") as f:
        lines = f.readlines()
    lines = [line.split(' ') for line in lines]
    img_idxs = set([line[0] for line in lines])
    for line in lines:
        cls_record[line[0]].append([float(x) for x in line[1].split(',')])

    for key in cls_record.keys():
        cls_record[key] = np.array(cls_record[key])
    # for img in img_idxs:
    #     cls_record[img] = np.array( [   list(map(int, line[1].split(',')))
    #                                     for line in lines if img==line[0]])

    return cls_record


def calc_iou(bbox, gt_bbox):
    """
    calculate iou between a predict bbox with ground truth bboxes
    input bbox: xyxy
    input gt_bbox: n*xyxy
    """
    # intersection
    xmin = np.maximum(gt_bbox[:, 0], bbox[0])
    ymin = np.maximum(gt_bbox[:, 1], bbox[1])
    xmax = np.minimum(gt_bbox[:, 2], bbox[2])
    ymax = np.minimum(gt_bbox[:, 3], bbox[3])
    w = np.maximum(xmax - xmin + 1.0, 0.0)
    h = np.maximum(ymax - ymin + 1.0, 0.0)
    inters = w * h
    # union
    uni = (
        (bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0) +
        (gt_bbox[:,2]-gt_bbox[:,0]+1.0) * (gt_bbox[:,3]-gt_bbox[:,1]+1.0)
        - inters
    )
    return inters / uni


def calc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    pred_cls_record: tuple,
    cls_name,
    iou_thresh=0.5,
    use_07_metric=False,
):
    """
    rec, prec, ap = voc_eval(...)
    input pred_cls_record: (img_name_array, data_array[xyxy, conf])
    Top level function that does the PASCAL VOC evaluation.
    """

    # extract ground truth objects for this class from test dataset
    cls_record = read_cls_anno(cls_name)
    num_positive = 0
    for data in cls_record.values():
        num_positive += sum(1 - data[:, -1])

    # parse predicted objects
    pred_imgs, pred_data = pred_cls_record[0], pred_cls_record[1]
    pred_conf = pred_data[:, 4]
    pred_bbox = pred_data[:, :4]
    # sort by confidence
    sorted_idx = np.argsort(-pred_conf)
    pred_imgs = pred_imgs[sorted_idx]
    pred_bbox = pred_bbox[sorted_idx]

    pred_len = len(pred_imgs)
    TP = np.zeros(pred_len)
    FP = np.zeros(pred_len)
    # for each predict bbox
    is_detected = {}
    for i, bbox in enumerate(pred_bbox):
        if len(cls_record[pred_imgs[i]])==0:  # whether img[i] is in record
            FP[i] = 1.0
            continue

        gt_bbox = cls_record[pred_imgs[i]][:, :4]
        gt_diff = cls_record[pred_imgs[i]][:, 4]

        if is_detected.get(pred_imgs[i]) is None: # create for every img in cls_rec
            is_detected[pred_imgs[i]] = np.zeros(len(gt_diff))

        ious = calc_iou(bbox, gt_bbox)
        iou_max = np.max(ious)
        max_idx = np.argmax(ious)

        if iou_max > iou_thresh:
            if not gt_diff[max_idx]:
                if not is_detected[pred_imgs[i]][max_idx]: # first time
                    TP[i] = 1.0
                    is_detected[pred_imgs[i]][max_idx] = 1
                else:
                    FP[i] = 1.0
        else:
            FP[i] = 1.0

    # compute precision and recall
    FP = np.cumsum(FP)
    TP = np.cumsum(TP)

    recall = TP / float(num_positive)
    # avoid divide by zero in case the first detection matches a difficult
    precision = TP / np.maximum(TP + FP, np.finfo(np.float64).eps)
    ap = calc_ap(recall, precision, use_07_metric)
    return ap


def voc_eval_ap50_95(
    pred_cls_record: tuple,
    cls_name,
    use_07_metric=False,
):
    """
    rec, prec, ap = voc_eval(...)
    input pred_cls_record: (img_name_array, data_array[xyxy, conf])
    Top level function that does the PASCAL VOC evaluation.
    """

    # extract ground truth objects for this class from test dataset
    cls_record = read_cls_anno(cls_name)
    num_positive = 0
    for data in cls_record.values():
        num_positive += sum(1 - data[:, -1])

    # parse predicted objects
    pred_imgs, pred_data = pred_cls_record[0], pred_cls_record[1]
    pred_conf = pred_data[:, 4]
    pred_bbox = pred_data[:, :4]
    # sort by confidence
    sorted_idx = np.argsort(-pred_conf)
    pred_imgs = pred_imgs[sorted_idx]
    pred_bbox = pred_bbox[sorted_idx]

    pred_len = len(pred_imgs)  # num of pred bboxes
    TP = np.zeros((pred_len, 10), dtype=np.float32)
    FP = np.zeros((pred_len, 10), dtype=np.float32)
    # for each predict bbox
    is_detected = {}
    for iou_threshold in np.arange(0.5, 1., 0.05):

        pass
    for i, bbox in enumerate(pred_bbox):
        if len(cls_record[pred_imgs[i]])==0:  # whether img[i] is in record
            FP[i] = 1.0
            continue

        gt_bbox = cls_record[pred_imgs[i]][:, :4]
        gt_diff = cls_record[pred_imgs[i]][:, 4]

        if is_detected.get(pred_imgs[i]) is None: # create for every img in cls_rec
            is_detected[pred_imgs[i]] = np.zeros((len(gt_diff), 10), dtype=np.int8)

        ious = calc_iou(bbox, gt_bbox)
        iou_max = np.max(ious)
        max_idx = np.argmax(ious)

        for iou_idx, iou_threshold in enumerate(np.arange(0.5, 1., 0.05)):
            if iou_max > iou_threshold:
                if not gt_diff[max_idx]:
                    if not is_detected[pred_imgs[i]][max_idx, iou_idx]:  # first time
                        TP[i, iou_idx] = 1.0
                        is_detected[pred_imgs[i]][max_idx, iou_idx] = 1
                    else:
                        FP[i, iou_idx] = 1.0
            else:
                FP[i, iou_idx] = 1.0

    # compute precision and recall
    FP = np.cumsum(FP, axis=0)
    TP = np.cumsum(TP, axis=0)

    recall = TP / float(num_positive)
    # avoid divide by zero in case the first detection matches a difficult
    precision = TP / np.maximum(TP + FP, np.finfo(np.float64).eps)

    ap50 = calc_ap(recall[:, 0], precision[:, 0], use_07_metric)
    ap50_95 = ap50
    for i in range(1, 10):
        ap50_95 += calc_ap(recall[:, i], precision[:, i], use_07_metric)

    return np.array([ap50, ap50_95/10.])

if __name__=='__main__':
    img_name = np.array(['5063','813','484'])
    data = np.array(
        [[1,2,3,4,0],
        [629,254,708,430,0],
        [12,43,57,87,0]], dtype=float
    )
    pred_cls_rec = (img_name, data)
    recall, precision, ap = voc_eval(pred_cls_rec, 'green_go')
    print(ap)


    # cls_rec = read_cls_anno('green_go')
    # print(cls_rec['813'])
