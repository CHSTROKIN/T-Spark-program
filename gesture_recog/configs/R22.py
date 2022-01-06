import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

os_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))


config = dict(
    cuda =  False,
    model_name = "mobilefacenet.MobileFaceNet",
    model_args = dict(),
    embed_dim = 512,
    num_classes = 22,
    class_fp = os.path.join(os_path,'gesture_recog/class-22.txt'),
    backbone_dir=os.path.join(os_path,'gesture_recog/models/R22/backbone-epoch10.pth'),
    head_dir=os.path.join(os_path,'gesture_recog/models/R22/head-epoch10.pth'),
    img_fp = os.path.join(os_path,'det_model/mac_1_cls0_0.jpg'),
    test_transform = transforms.Compose([
        transforms.Resize([224, 224], interpolation=Image.BILINEAR),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [1, 1, 1])
    ])
)
