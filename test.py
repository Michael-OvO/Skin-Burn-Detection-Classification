import argparse
import json
import os
import time
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, increment_path, set_logging, \
    box_iou, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized, TracedModel


def test(
        data,
        weights=None,
        batch_size=32,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        save_json=False,
        single_cls=False,
        augment=False,
        verbose=False,
        model=None,
        dataloader=None,
        save_dir=Path('.'),
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        plots=True,
        wandb_logger=None,
        compute_loss=None,
        half_precision=True,
        trace=False,
        is_coco=False,
        ):
    """Run validation.

    Returns tuple of (results, maps, times)
    """
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        set_logging()
        device = select_device('', batch_size=batch_size)
        save_dir = Path(increment_path(Path('runs/test') / 'exp', exist_ok=False))
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        model = attempt_load(weights, map_location=device)
        stride = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=stride)
        if trace:
            model = TracedModel(model, device, imgsz)

    half = device.type != 'cpu' and half_precision
    if half:
        model.half()

    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        task = 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, None, pad=0.5, rect=True)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    t0, t1 = 0., 0.
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    jdict = []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape

        with torch.no_grad():
            t = time_synchronized()
            out, train_out = model(img, augment=augment)
            t0 += time_synchronized() - t

            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]

            targets[:, 2:] *= torch.tensor([width, height, width, height], device=device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            if save_txt:
                gn = torch.tensor(shapes[si][0], device=device)[[1, 0, 1, 0]]
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    with open(save_dir / 'labels' / f'{path.stem}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))
                for cls in torch.unique(labels[:, 0]):
                    ti = (cls == labels[:, 0]).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected:
                                detected.append(d.item())
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)
        mp = mr = map50 = map = 0.0
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    t = (t0 / seen * 1E3, t1 / seen * 1E3, imgsz, imgsz, batch_size)
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    model.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report AP per class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrids')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in labels')
    parser.add_argument('--save-json', action='store_true', help='save JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--no-trace', action='store_true', help='don"t trace model')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)
    print(opt)
    if opt.task in ('train', 'val', 'test'):
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             )
    elif opt.task == 'speed':
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False)
    elif opt.task == 'study':
        x = list(range(256, 1536 + 128, 128))
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'
            y = []
            for i in x:
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json, plots=False)
                y.append(r + t)
            np.savetxt(f, y, fmt='%10.4g')
        os.system('zip -r study.zip study_*.txt')
