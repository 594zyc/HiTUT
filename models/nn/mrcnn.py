import os, io, sys
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image

from gen.constants import *
import models.detector.transforms as T
from models.detector.mrcnn import load_pretrained_model


class MaskRCNNDetector(nn.Module):

    def __init__(self, args, detectors=[]):
        super().__init__()
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.topk = args.topk_objs
        self.confidence = args.confidence

        self.transform = T.Compose([T.ToTensor()])   # img to Tensor

        self.eval_detector = args.detector_type

        if 'all' in detectors:
            # all at once detector
            self.classes = ACTION_ARGS
            self.model = load_pretrained_model(args.all_detector_path,
                len(self.classes), device=self.device)
            self.model.to(self.device)
            self.model.eval()
        if 'sep' in detectors:
            # movable object detector
            self.classes_obj = OBJECTS_DETECTOR + ['None']
            self.model_obj = load_pretrained_model(args.obj_detector_path,
                len(self.classes_obj), device=self.device)
            self.model_obj.to(self.device)
            self.model_obj.eval()

            # body = self.model_obj.backbone.body
            # print(body)
            # for layer in body.named_modules():
            #     print(layer)
            # a = torch.rand((1,3,300,300)).to(self.device)
            # feature = body(a)
            # print(feature)
            # for k,v in feature.items():
            #     print(k, v.shape)

            # static receptacle detector
            self.classes_rec = STATIC_RECEPTACLES + ['None']
            self.model_rec = load_pretrained_model(args.rec_detector_path,
                len(self.classes_rec), device=self.device)
            self.model_rec.to(self.device)
            self.model_rec.eval()

    def get_preds_step(self, frame):
        """ get detection results for a single step

        Args:
            frame: current visual observation returned by iTHOR
                        an uint8 np array of size w x h x 3(BGR)

        Returns:
            list of masks, boxes, classes, scores
        """
        if self.eval_detector == 'all':
            masks, boxes, classes, scores = self.get_mrcnn_preds_all([frame])
        elif self.eval_detector == 'sep':
            masks, boxes, classes, scores = self.get_mrcnn_preds_sep([frame])
        return masks[0], boxes[0], classes[0], scores[0]


    def get_mrcnn_preds_all(self, img_batch):
        """
        get_prediction
          parameters:
            - img - path of the input image or frame from AI2THOR
            - confidence - threshold to keep the prediction or not
          method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
              ie: eg. segment of cat is made 1 and rest of the image is made 0

        """
        img_input = []
        for idx, img in enumerate(img_batch):
            if isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, bytes):
                img = Image.open(io.BytesIO(np.array(img)))
            else:
                img = Image.fromarray(frame)
                img.show()
                quit()
            img, _ = self.transform(img.convert("RGB"), None)
            img_input.append(img.to(self.device))

        predictions = self.model(img_input)

        masks_col, pred_boxes_col, pred_class_col, score_col = [], [], [], []
        for idx, pred in enumerate(predictions):

            pred_score = list(pred['scores'].detach().cpu().numpy())
            # pruned predictions with a confidence score lower than the set value
            pred_pruned = [pred_score.index(x) for x in pred_score if x>self.confidence]

            # if len(pred_pruned) >= self.topk:
            #     print(img_batch[idx])
            #     print('topk:', self.topk, 'detected:', len(pred_pruned))
            # keep the only topk detections
            pred_pruned = pred_pruned[:self.topk]

            if len(pred_pruned) > 0:
                pred_t = pred_pruned[-1]
                masks = (pred['masks']>0.5).squeeze().detach().cpu().numpy()
                pred_class = [self.classes[i] for i in list(pred['labels'].detach().cpu().numpy())]
                pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred['boxes'].detach().cpu().numpy())]
                if len(masks.shape) == 2:
                    masks = masks[np.newaxis, ...]
                else:
                    masks = masks[:pred_t+1]
                pred_boxes = pred_boxes[:pred_t+1]
                pred_class = pred_class[:pred_t+1]
                pred_score = pred_score[:pred_t+1]
            else:
                masks, pred_boxes, pred_class, pred_score = [], [], [], []

            masks_col.append(masks)
            pred_boxes_col.append(pred_boxes)
            pred_class_col.append(pred_class)
            score_col.append(pred_score)
        return masks_col, pred_boxes_col, pred_class_col, score_col


    def get_mrcnn_preds_sep(self, img_batch):
        """
        get_prediction
          parameters:
            - img - path of the input image or frame from AI2THOR
            - confidence - threshold to keep the prediction or not
          method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
              ie: eg. segment of cat is made 1 and rest of the image is made 0

        """
        img_input = []
        for idx, img in enumerate(img_batch):
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, bytes):
                img = Image.open(io.BytesIO(np.array(img))).convert('RGB')
            else:
                img = Image.fromarray(img)
            img, _ = self.transform(img, None)
            img_input.append(img.to(self.device))

        predictions_obj = self.model_obj(img_input)
        predictions_rec = self.model_rec(img_input)

        masks_col, pred_boxes_col, pred_class_col, score_col = [], [], [], []
        for m, predictions in enumerate([predictions_obj, predictions_rec]):
            class_map = self.classes_obj if m == 0 else self.classes_rec
            for idx, pred in enumerate(predictions):
                pred_score = list(pred['scores'].detach().cpu().numpy())
                # pruned predictions with a confidence score lower than the set value
                pred_pruned = [pred_score.index(x) for x in pred_score if x>self.confidence]

                # keep the only topk detections
                pred_pruned = pred_pruned

                if len(pred_pruned) > 0:
                    pred_t = pred_pruned[-1]
                    masks = (pred['masks']>0.5).squeeze().detach().cpu().numpy()
                    pred_class = [class_map[i] for i in list(pred['labels'].detach().cpu().numpy())]
                    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred['boxes'].detach().cpu().numpy())]
                    if len(masks.shape) == 2:
                        masks = masks[np.newaxis, ...]
                    else:
                        masks = masks[:pred_t+1]
                    pred_boxes = pred_boxes[:pred_t+1]
                    pred_class = pred_class[:pred_t+1]
                    pred_score = pred_score[:pred_t+1]
                else:
                    masks, pred_boxes, pred_class, pred_score = [], [], [], []

                if m == 0:
                    masks_col.append(masks)
                    pred_boxes_col.append(pred_boxes)
                    pred_class_col.append(pred_class)
                    score_col.append(pred_score)
                else:
                    if masks_col[idx] == []:
                        masks_col[idx] = masks
                    elif masks != []:
                        masks_col[idx] = np.concatenate([masks_col[idx], masks], axis=0)
                    pred_boxes_col[idx] += pred_boxes
                    pred_class_col[idx] += pred_class
                    score_col[idx] += pred_score

        for img_idx, scores in enumerate(score_col):
            sorted_idx = [i[0] for i in sorted(enumerate(scores), key=lambda x: -x[1])]
            score_col[img_idx] = [score_col[img_idx][i] for i in sorted_idx][:self.topk]
            masks_col[img_idx] = [masks_col[img_idx][i] for i in sorted_idx][:self.topk]
            pred_boxes_col[img_idx] = [pred_boxes_col[img_idx][i] for i in sorted_idx][:self.topk]
            pred_class_col[img_idx] = [pred_class_col[img_idx][i] for i in sorted_idx][:self.topk]
        return masks_col, pred_boxes_col, pred_class_col, score_col
