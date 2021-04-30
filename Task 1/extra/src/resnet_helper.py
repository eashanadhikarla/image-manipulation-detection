import os
import argparse

import json
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms as T
from torchvision import datasets, models, transforms
import torch, torchvision, torch.nn as nn, torch.optim as optim
from torchvision.datasets.utils import download_url

with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

model = models.resnext50_32x4d(pretrained=True)
device = torch.device("cpu")


def preprocess(IMAGE):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = image_transforms(IMAGE)
    return img


def classify_resnet(img):
    x = cv2.resize(img, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = Image.fromarray(x)
    x = preprocess(x)
    x = x.unsqueeze_(0)
    x = x.float()

    output = model.forward(x)
    # preds  = torch.topk(output, 3)
    _, indices = torch.sort(output, descending=True)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    preds = [(classes[idx], percentage[idx].item()) for idx in indices[0][:3]]
    # print(preds[0][0], preds[0][1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    linepos = [12, 24, 36]

    # print top 3 results
    for i in range(3):
        cv2.putText(img, '{}: {:.3f}'.format(preds[i][0], preds[i][1]),
                    (10, linepos[i]),
                    font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return img

# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np
# import cv2

# model = ResNet50(weights='imagenet')

# def classify_resnet(img):
#     x = cv2.resize(img, (224, 224))
#     x = image.img_to_array(x)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     preds = model.predict(x)
#     dec_preds = decode_predictions(preds, top=3)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     linepos = [12, 24, 36]

#     # print top 3 results
#     for i in range(3):
#         cv2.putText(img, '{}: {:.3f}'.format(dec_preds[0][i][1],
#                                              dec_preds[0][i][2]),
#                     (10, linepos[i]),
#                     font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

#     return img
