"""Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path
from mindspore.train.serialization import save_checkpoint
import numpy as np
import yaml
from tqdm import tqdm
from mindspore.context import ParallelMode
from mindspore import Tensor, context, nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore import dtype as ms
from models.yolov5s import Model, YoloWithLossCell, TrainingWrapper
from train_utils.yolo_dataset import create_dataloader
from train_utils.general import labels_to_class_weights, check_img_size, colorstr, fitness

PROJ_DIR = Path(__file__).resolve().parent   # TODO 项目根目录，以下所有目录相对于此


def train(hyp,  opt):
    # opt
    epochs, batch_size, weights, data, noval, nosave= \
        opt.epochs, opt.batch_size, opt.weights, opt.data, opt.noval, opt.nosave
    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device)  #change "GPU" when needed
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    # Directories
    save_dir = Path(PROJ_DIR / opt.save_dir)
    weight_dir = save_dir / 'weights'
    weight_dir.mkdir(parents=True, exist_ok=True)  # make dir
    last = weight_dir / 'last.pt'
    best = weight_dir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Image sizes
    gs = 32  # grid size (max stride)
    nl = 3  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_val = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        opt.weights = str(opt.weights)
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # datasest config
    with open(data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check
    train_path = r'D:\Huawei\speedstar\data\datav4\test.txt'
    val_path = data_dict['val']

    # 训练集
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, stride=gs, hyp=hyp, augment=True, cache=opt.cache_images,
                      rect=opt.rect, workers=opt.workers, image_weights=opt.image_weights, multi_process=opt.multi_process)

    nb = len(dataset) // batch_size # number of batches

    # 测试集 TODO
    # valloader = create_dataloader(val_path, imgsz_val, batch_size * 2, gs, single_cls,
    #                                   hyp=hyp, cache=opt.cache_images and not noval, rect=True, rank=-1,
    #                                   workers=workers,
    #                                   pad=0.5, prefix=colorstr('val: '))[0]

    # --- Model
    if weights.suffix=='.ckpt' and opt.resume:  # 如果给了预训练模型，并且resume，加载预训练模型
        # TODO 加载模型
        param_dict = load_checkpoint(str(weights))
        img_size = param_dict['module1_8.conv2d_0.weight'].shape[1]
        assert img_size==imgsz, 'image size of weights conflict!'
        model = Model(bs=batch_size, img_size=imgsz)  # batch size
        not_load_params = load_param_into_net(model, param_dict)
        model.set_train(True)
    else:   # 初始化模型
        model = Model(bs=batch_size, img_size=imgsz)  # create
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc) * nc  # attach class weights
    model.names = names

    # --- Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")
    # 学习率
    lr = nn.dynamic_lr.cosine_decay_lr(0.0, hyp['lr0'], nb*opt.epochs, nb, 5)
    if opt.adam:
        optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr,
                            betas1=hyp['momentum'], weight_decay=hyp['weight_decay'])
    else:
        optimizer = nn.SGD(params=model.trainable_params(), learning_rate=lr,
                            momentum=hyp['momentum'], nesterov=True, weight_decay=hyp['weight_decay'])

    # --- connect
    model_train = TrainingWrapper(YoloWithLossCell(model), optimizer, accumulate)
    model_train.set_train()

    # --- Start training
    t0 = time.time()
    n_warm = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    mAPs = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    start_epoch = 0
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        print(('\n' + '%10s' * 3) % ('Epoch', 'loss', 'img_size'))
        mloss = 0.0  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)
        for i, data in pbar:  # batch -------------------------------------------------
            imgs = Tensor.from_numpy(data["images"].astype(np.float) / 255.0).astype(ms.float32) # uint8 to float32, 0-255 to 0.0-1.0
            batch_y_true_0 = Tensor.from_numpy(data['bbox1']).astype(ms.float32)
            batch_y_true_1 = Tensor.from_numpy(data['bbox2']).astype(ms.float32)
            batch_y_true_2 = Tensor.from_numpy(data['bbox3']).astype(ms.float32)
            batch_gt_box0 = Tensor.from_numpy(data['gt_box1']).astype(ms.float32)
            batch_gt_box1 = Tensor.from_numpy(data['gt_box2']).astype(ms.float32)
            batch_gt_box2 = Tensor.from_numpy(data['gt_box3']).astype(ms.float32)
            # forward
            loss = model_train(imgs, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                       batch_gt_box2)

            # Print
            mloss = (mloss * i + loss) / (i + 1)  # update mean losses
            s = ('%10s' + '%10.4g' * 2) % (
                f'{epoch}/{epochs - 1}', mloss.asnumpy(), imgs.shape[-1])
            pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------

        # mAP TODO
        # Save model
        if epoch % 10 == 0:
            save_checkpoint(model, 'runs/exp/weights')

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    print(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # option file
    parser.add_argument('--option', type=str, default='./opt.yaml', help='option.yaml path')
    # train config
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    # read option file
    opt_file = PROJ_DIR / opt.option
    assert opt_file.exists(), f"opt yaml({str(opt_file)}) does not exist"
    with opt_file.open('r') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace

    opt.multi_process = (opt.device == 'GPU')
    # check weights file
    opt.weights = PROJ_DIR / opt.weights
    assert opt.weights.exists(), f'ERROR: resume weights({str(opt.weights)}) checkpoint does not exist'
    print('Resuming training from %s' % str(opt.weights))
    opt.data = str(PROJ_DIR / opt.data)
    opt.hyp = str(PROJ_DIR / opt.hyp)
    # print
    print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    return opt


def main(opt):
    # Train
    train(opt.hyp, opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
