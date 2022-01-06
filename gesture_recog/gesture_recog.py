import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os, sys, time, argparse, shutil, importlib, random
from os.path import join, abspath, dirname, isfile, split, splitext
from PIL import Image, ImageFont, ImageDraw
import numpy as np

from configs import get_default
from backbone import mobilefacenet
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152, ResNet_18
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from efficientnet_pytorch.model import EfficientNet
import cv2
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")) , 'gesture_recog'))


# parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
# parser.add_argument("--config", type=str, default=os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")) , 'gesture_recog/configs/handpose_20210611.py'))
# args = parser.parse_args()

# # load configs
# spec = importlib.util.spec_from_file_location("module.name", args.config)
# config_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(config_module)
# config = config_module.config

# # set configs
# config["embed_dim"] = config["embed_dim"] if "embed_dim" in config else 512
rec_data = None

def rec_init(parser):
	args = parser.parse_args()
	# load configs
	spec = importlib.util.spec_from_file_location("module.name", args.config)
	config_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(config_module)
	config = config_module.config
	# set configs
	config["embed_dim"] = config["embed_dim"] if "embed_dim" in config else 512
	isCuda = config['cuda']
	if isCuda:
		torch.backends.cudnn.benchmark = True
	classes = read_class(config["class_fp"])
	transform = config["test_transform"]
	model_name = eval(config["model_name"])
	backbone = model_name(**config["model_args"])
	num_classes = config["num_classes"]
	head = nn.Linear(config["embed_dim"], num_classes)
	if os.path.isfile(config["backbone_dir"]) and os.path.isfile(config["head_dir"]):
		backbone.load_state_dict(torch.load(config["backbone_dir"], map_location=torch.device('cpu')))
		head.load_state_dict(torch.load(config["head_dir"], map_location=torch.device('cpu')))
	else:
		print('backbone_dir or head_dir not exist.')
	global rec_data
	backbone.eval()
	head.eval()
	rec_data = [backbone, head, classes, transform]


def rec_hand(img_):
	global rec_data
	if img_ is None: return None
	backbone, head, classes, transform = rec_data
	im = Image.fromarray(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
	im = transform(im)
	c, w, h = im.shape
	im = im.view(-1, c, w, h)
	# recog
	topk = (1,)
	with torch.no_grad():
		feature = backbone(im)
		output = head(feature)
	maxk = max(topk)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	isCuda = False
	if isCuda:
		pred = pred.cpu().numpy()[0][0]
	else:
		pred = pred.numpy()[0][0]
	# To prevent errors
	if pred in classes:
		cls = classes[pred]
	else:
		cls = "None"
	return cls

def rec_hands(crops):
	cls_list = []
	for crop in crops:
		cls_list.append(rec_hand(crop))
	return cls_list


def gesture_recog(backbone, head, im, isCuda=False):
    if isCuda:
        device = torch.device("cuda:0")
        backbone = backbone.to(device)
        head = head.to(device)
        im = im.to(device)

    backbone.eval()
    head.eval()
    topk = (1,)

    with torch.no_grad():
        feature = backbone(im)
        output = head(feature)
    
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if isCuda:
        pred = pred.cpu().numpy()[0][0]
    else:
        pred = pred.numpy()[0][0]
    return pred

def read_class(class_fp):
    classes = {}
    lines = open(class_fp).readlines()
    for l in lines:
        sp = l.strip().split(' ')
        classes[int(sp[1])] = sp[0]
    return classes

if __name__ == '__main__':
    # init 
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
    parser.add_argument("--config", type=str, default=os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")) , 'gesture_recog/configs/example-cls3.py'))
    args = parser.parse_args()

    # load configs
    spec = importlib.util.spec_from_file_location("module.name", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config

    # set configs
    config["embed_dim"] = config["embed_dim"] if "embed_dim" in config else 512

    isCuda = config['cuda']
    if isCuda:
        torch.backends.cudnn.benchmark = True

    classes = read_class(config["class_fp"])

    model_name = eval(config["model_name"])
    backbone = model_name(**config["model_args"])

    num_classes = config["num_classes"]
    head = nn.Linear(config["embed_dim"], num_classes)
    if os.path.isfile(config["backbone_dir"]) and os.path.isfile(config["head_dir"]):
        backbone.load_state_dict(torch.load(config["backbone_dir"], map_location=torch.device('cpu')))
        head.load_state_dict(torch.load(config["head_dir"], map_location=torch.device('cpu')))
    else:
        print('backbone_dir or head_dir not exist.')

    im = Image.open(config["img_fp"])
    transform = config["test_transform"]
    im = transform(im)
    c, w, h = im.shape
    im = im.view(-1, c, w, h)

    pred = gesture_recog(backbone, head, im, isCuda=isCuda)
    cls = classes[pred]
    print('This image is gesture '+cls+'!')
