# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import torch
import argparse
import shutil
from PIL import Image

import cv2

from engine import train_one_epoch, evaluate
import utils
import torchvision
import transforms as T
from mrcnn import get_model_instance_segmentation, load_pretrained_model
from train_mrcnn import AlfredDataset, get_transform, get_object_classes

import sys
sys.path.insert(0, os.environ["ALFRED_ROOT"])
import gen.constants as constants

os.system("taskset -p 0xffffffff %d" % os.getpid())
# sys.stdout = open('eval_result', 'a')

MIN_PIXELS = 100

OBJECTS_DETECTOR = constants.OBJECTS_DETECTOR
STATIC_RECEPTACLES = constants.STATIC_RECEPTACLES
ALL_DETECTOR = constants.ALL_DETECTOR


def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = len(get_object_classes(args.object_types))+1
    # use our dataset and defined transformations
    dataset_test = AlfredDataset(args.data_path, get_transform(train=False), args)

    # define training and validation data loaders
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = load_pretrained_model(args.load_model, num_classes)

    # move model to the right device
    model.to(device)

    # evaluate
    evaluate(model, data_loader_test, args, device=device)

    print("Done testing!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="detector/data/test_unseen")
    parser.add_argument("--object_types", choices=["objects", "receptacles", "all"], default="objects")
    parser.add_argument("--save_name", type=str, default="mrcnn_alfred_objects")
    parser.add_argument("--load_model", type=str, default="agents/detector/models/mrcnn.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_num", type=int, default=10000)
    parser.add_argument("--balance_scenes", action='store_true')
    parser.add_argument("--save_image", action='store_true')
    parser.add_argument("--show_image", action='store_true')
    parser.add_argument("--save_path", type=str, default="detector/examples_unseen")



    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(os.path.join(args.save_path, 'gt'))
    os.makedirs(os.path.join(args.save_path, 'pred'))

    # print(args.data_path, args.object_types, args.load_model, args.test_num)

    main(args)
