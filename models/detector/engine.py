import math
import sys, os
import time
import torch
import glob
import cv2
import numpy as np
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    # count = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # count += 1
        # if count % 100 == 0:
        #     torch.cuda.empty_cache()


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, args, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(n_threads)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    coco = get_coco_api_from_dataset(data_loader.dataset)
    object_classes = data_loader.dataset.object_classes
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
        image = list(img.to(cpu_device) for img in image)

        if args.save_image:
            for i, img in enumerate(image):
                gt = targets[i]
                pred = outputs[i]
                path_gt = os.path.join(args.save_path, 'gt')
                save_visualization(img, gt, object_classes, path_gt, show_image=args.show_image)
                path_pd = os.path.join(args.save_path, 'pred')
                save_visualization(img, pred, object_classes, path_pd, threshold=0.7, show_image=args.show_image)

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator



def save_visualization(image, target, object_classes, save_path, threshold=None, show_image=False):
    # image = transforms.ToPILImage(image)
    # print(image)
    # print(image.shape)
    disp_img = np.array(image.mul(255).byte().permute(1, 2, 0))
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    if len(target['boxes']) == 0:
        im_ind = get_image_index(save_path)
        cv2.imwrite(save_path + '/%05d_image.png' % im_ind, disp_img)
        sg_merge = np.uint8(torch.ones_like(disp_img[:, :, 0]).mul(255))
        cv2.imwrite(save_path + '/%05d_masks.png' % im_ind, sg_merge)
        return
    sg_merge = torch.zeros_like(target['masks'][0].squeeze()[:, :, np.newaxis])
    for idx in range(len(target['boxes'])):
        xmin, ymin, xmax, ymax = map(int, target['boxes'][idx])
        smask = target['masks'][idx]
        object_class = object_classes[target['labels'][idx]]
        if threshold is not None and target['scores'][idx] < threshold:
            continue

        cv2.rectangle(disp_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.putText(disp_img, object_class, (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), thickness=1)
        # sg = np.uint8(smask[:, :, np.newaxis]) * int(target['labels'][idx].item())
        sg = smask.squeeze()[:, :, np.newaxis]
        sg_merge += sg
    sg_merge[sg_merge>0.5] = 1
    sg_merge = np.uint8(sg_merge.mul(255))

    if show_image:
        cv2.imshow("img", disp_img)
        cv2.imshow("sg", sg_merge)
        cv2.waitKey(0)

    im_ind = get_image_index(save_path)
    cv2.imwrite(save_path + '/%05d_image.png' % im_ind, disp_img)
    cv2.imwrite(save_path + '/%05d_masks.png' % im_ind, sg_merge)


def get_image_index(save_path):
    return len(glob.glob(save_path + '/*_image.png'))