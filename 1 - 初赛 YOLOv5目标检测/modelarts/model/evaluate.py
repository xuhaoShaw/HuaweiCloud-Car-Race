
from eval.evaluator import Evaluator
import torch
import yolov5_config as cfg
import numpy as np

if __name__ == '__main__':
    import time
    from models.yolo import Model

    # load model
    print("loading weight file from : {}".format(cfg.EVAL_WEIGHT_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cfg.EVAL_WEIGHT_PATH, map_location=device)
    # print(ckpt['model'].yaml)
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


