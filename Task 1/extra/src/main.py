import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms as T
from torchvision import datasets, models, transforms
import torch, torchvision, torch.nn as nn, torch.optim as optim
from torchvision.datasets.utils import download_url

import argparse
from resnet_helper import *

# from vgg_helper import *

ap = argparse.ArgumentParser()
ap.add_argument('--classifier', help='Classifier must be "resnet" or "vgg"')
args = ap.parse_args()


def camera_loop():
    cap = cv2.VideoCapture(0)
    while (True):
        _, frame = cap.read()

        action = cv2.waitKey(1)

        if args.classifier == 'resnet':
            # resnet prediction
            frame = classify_resnet(frame)
        # elif args.classifier == 'vgg':
        #     # VGG prediction
        #     frame = classify_vgg(frame)
        cv2.imshow('camera', frame)

        if action == ord('q') or action == 27:
            break

        if action == ord('r'):
            # resnet prediction
            frame = classify_resnet(frame)
            cv2.imshow('ResNet', frame)
        # if action == ord('v'):
        #     # vgg prediction
        #     frame = classify_vgg(frame)
        #     cv2.imshow('VGG', frame)

    cap.release()


if __name__ == '__main__':
    camera_loop()
    cv2.destroyAllWindows()
