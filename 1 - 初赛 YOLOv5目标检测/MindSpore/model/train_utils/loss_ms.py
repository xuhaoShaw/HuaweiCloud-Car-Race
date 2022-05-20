# Loss functions
import mindspore
from general import colorstr
import numpy as np
from mindspore import nn, Tensor, Model, dtype, context, Parameter
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore as ms
import mindspore.ops as ops
import math
ms.common.set_seed(0)

def clamp(input: Tensor, min=None, max=None):
    if min:
        input[input < min] = min
    if max:
        input[input > max] = max
    return input

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps



class BCEBlurWithLogitsLoss(nn.Cell):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.sigmoid = ops.Sigmoid()
        self.exp = ops.Exp()

    def construct(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = self.sigmoid(pred) # prob from logits
        dx = pred - true  # reduce only missing label effects
        alpha_factor = 1 - self.exp( (dx - 1) / (self.alpha + 1e-4) )
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.bce_with_logits_loss.reduction
        self.loss_fcn.bce_with_logits_loss.reduction = 'none'  # required to apply FL to each element
        self.sigmoid = ops.Sigmoid()

    def construct(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = self.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Cell):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.bce_with_logits_loss.reduction
        self.loss_fcn.bce_with_logits_loss.reduction = 'none'  # required to apply FL to each element
        self.sigmoid = ops.Sigmoid()
        self.abs = ops.Abs()

    def construct(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = self.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = self.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss(nn.Cell):
    # Compute losses
    def __init__(self, model, autobalance=False):  #TODO model.hyp
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=Tensor([h['cls_pw']], dtype.float32))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=Tensor([h['obj_pw']], dtype.float32))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.balance = Parameter(Tensor([4.0, 1.0, 0.4], dtype.float32), requires_grad=False) #{3: [4.0, 1.0, 0.4]}.get(3, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = 1 if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.na = 3 # num of anchors
        self.nc = 6 # num of class
        self.nl = 3 # num of layers
        self.anchors = Tensor([[[ 1.25000,  1.62500],
                                [ 2.00000,  3.75000],
                                [ 4.12500,  2.87500]],

                                [[ 1.87500,  3.81250],
                                [ 3.87500,  2.81250],
                                [ 3.68750,  7.43750]],

                                [[ 3.62500,  2.81250],
                                [ 4.87500,  6.18750],
                                [11.65625, 10.18750]]], dtype=dtype.float32)
        self.sig = ops.Sigmoid()
        self.cat1 = ops.Concat(1)
        self.cat0 = ops.Concat(0)
        self.cat2 = ops.Concat(2)
        self.argsort = lambda x: Tensor(x.asnumpy().argsort())
        self.ones = ops.Ones()
        self.repeat = ops.Tile()
        self.max = ops.Maximum()
        self.stack = ops.Stack()
        self.logical_and = ops.LogicalAnd()
        self.min, self.pow, self.atan = ops.Minimum(), ops.Pow(), ops.Atan()
        self.reduce_max = ops.ReduceMax()
        self.range = ops.Range()

    def construct(self, p, targets):  # predictions, targets, model
        lcls, lbox, lobj = Tensor(0, dtype.float32), Tensor(0, dtype.float32), Tensor(0, dtype.float32)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = ops.zeros_like(pi[..., 0])  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = self.sig(ps[:, :2]) * 2. - 0.5
                pwh = (self.sig(ps[:, 2:4]) * 2) ** 2 * anchors[i]
                pbox = self.cat1((pxy, pwh))  # predicted box
                iou = self.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = clamp(iou, 0).astype(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = self.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = ps[:, 5:].fill(self.cn).astype(dtype.float32)
                    # t = Tensor.from_numpy(np.full_like(ps[:, 5:], self.cn, np.float)).astype(dtype.float32)
                    t[list(range(n)), tcls[i]] = Tensor(self.cp, dtype.float32)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.item()

        if self.autobalance:
            for i in len(self.balance):
                self.balance[i] = self.balance[i] / self.balance[self.ssi]
            # self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, self.cat0((lbox, lobj, lcls, loss))

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = self.ones(7, dtype.float32)  # normalized to gridspace gain
        ####
        ai = self.repeat(self.range(0.0, na.astype(dtype.float32), 1.0).view(na, 1), (1, nt))  # same as .repeat_interleave(nt)
        targets = self.cat2((self.repeat(targets, (na, 1, 1)),  ai[:, :, None]))  # append anchor indices

        g = 0.5  # bias
        off = Tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], dtype=dtype.float32) * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = Tensor(p[i].shape, dtype.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = self.reduce_max(self.max(r, 1. / r), 2) < self.hyp['anchor_t']
                # j = Tensor.from_numpy(self.max(r, 1./r).asnumpy().max(2)) < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = Tensor(t.asnumpy()[j.asnumpy()], t.dtype)  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = self.logical_and((gxy % 1. < g), (gxy > 1.)).T
                l, m = self.logical_and((gxi % 1. < g), (gxi > 1.)).T
                j = self.stack((j.fill(True), j, k, l, m)).asnumpy()
                t = Tensor(self.repeat(t, (5, 1, 1)).asnumpy()[j], dtype=t.dtype)
                offsets = Tensor((ops.zeros_like(gxy)[None] + off[:, None]).asnumpy()[j], off.dtype)
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(dtype.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(dtype.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(dtype.int64)  # anchor indices
            indices.append((b, a, clamp(gj, 0, gain[3].asnumpy().item() - 1), clamp(gi, 0, gain[2].asnumpy().item() - 1)))   # image, anchor, grid indices
            tbox.append(self.cat1((gxy - gij, gwh)))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T
        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area

        inter = clamp(self.min(b1_x2, b2_x2) - self.max(b1_x1, b2_x1), 0) * \
                clamp(self.min(b1_y2, b2_y2) - self.max(b1_y1, b2_y1), 0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = self.max(b1_x2, b2_x2) - self.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = self.max(b1_y2, b2_y2) - self.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if DIoU:
                    iou = iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * self.pow(self.atan(w2 / h2) - self.atan(w1 / h1), 2)
                    #with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                    iou = iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou = iou - (c_area - union) / c_area  # GIoU

        return iou  # IoU

class ModelWithLoss(nn.Cell):
    """
    connect model with loss
    """
    def __init__(self, model, loss_fn):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
    def construct(self, imgs, label):
        pred = self.model(imgs)
        loss, loss_item = self.loss_fn(pred[0:3])
        return loss, loss_item

class TrainingWrapper():
    """Training wrapper."""

    def __init__(self, network, optimizer, accumulate=1, sens=1.0):
        super(TrainingWrapper, self).__init__()
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        # accumulate grads
        self.accumulate = accumulate
        self.iter_cnt = 0
        self.acc_grads = None

    def __call__(self, imgs, labels):
        self.iter_cnt += 1
        weights = self.weights
        loss, loss_items = self.network(imgs, labels)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(imgs, labels, sens)
        # accumulate grads
        if self.iter_cnt == 1: # 积累周期开始
            self.acc_grads = list(grads)
        else: # 积累
            for i in range(len(grads)):
                self.acc_grads[i] += grads[i]
        if self.iter_cnt % self.accumulate == 0:  # 更新
            self.iter_cnt = 0
            if self.reducer_flag:
                self.acc_grads = self.grad_reducer(self.acc_grads)
            self.optimizer(self.acc_grads)
        return loss, loss_items

# test
if __name__ == '__main__':
    import sys
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    sys.path.append(str(BASE_DIR.joinpath('..').resolve()))
    from models.yolov5s import Model
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    import yaml
    from train_utils.general import labels_to_class_weights
    # model
    model = Model(img_size=1024, bs=8)
    hpy_file = BASE_DIR / '../../../data/hyps/hyp.speedstar.yaml'
    with open(hpy_file) as f:
        hyp = yaml.safe_load(f)
    nl, nc, imgsz = 3, 6, 1024
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = Tensor([1.0007, 1.2161, 0.83789, 0.97289, 0.76471, 1.2077], dtype.float32)
    model.names = ['red_stop', 'green_go', 'yellow_back', 'speed_limited', 'speed_unlimited', 'pedestrian_crossing']

    compute_loss = ComputeLoss(model)
    targets = Tensor([[0.00000, 0.00000, 0.08701, 0.66769, 0.04519, 0.08280],
        [0.00000, 3.00000, 0.44500, 0.72020, 0.02538, 0.03608],
        [1.00000, 0.00000, 0.43246, 0.70731, 0.02414, 0.05162]], dtype.float32)
    preds = (Tensor(np.ones([2, 3, 128, 128, 11], dtype=np.float32)),
            Tensor(np.ones([2, 3, 64, 64, 11], dtype=np.float32)),
            Tensor(np.ones([2, 3, 32, 32, 11], dtype=np.float32)))
    loss, loss_items = compute_loss(preds, targets)
    print(loss)
    print(loss_items)
    from mindspore import Model as m
