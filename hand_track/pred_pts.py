from hand_track.models.resnet import resnet18, resnet34, resnet50, resnet101
from hand_track.models.squeezenet import squeezenet1_1, squeezenet1_0
from hand_track.models.shufflenetv2 import ShuffleNetV2
from hand_track.models.shufflenet import ShuffleNet
from hand_track.models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0
from hand_track.models.rexnetv1 import ReXNetV1
from hand_track.models.resnet12 import ResNet12
from hand_track.models.mini_vgg import MiniVGG

from hand_track.hand_data_iter.datasets import draw_bd_handpose

import torch
import torch.nn as nn

import os, sys, time, argparse, shutil, importlib, random
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# 摄像头
import cv2
# 项目路径
proj_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

# 导入识别库路径
sys.path.append(os.path.join(proj_path, 'gesture_recog'))
from configs import get_default
from backbone import mobilefacenet

def pad_img(im, x, y, w, h, scale):
	cx = x + w / 2
	cy = y + h / 2
	boxLen = max(w, h)
	invXOffset = cx - boxLen * scale * 0.5
	invYOffset = cy - boxLen * scale * 0.5
	M = [[1.0, 0, -invXOffset], [0, 1.0, -invYOffset]]
	M = np.array(M)
	im = cv2.warpAffine(im, M, (int(boxLen * scale), int(boxLen * scale)), flags=cv2.INTER_LINEAR)
	return im, invXOffset, invYOffset


def align_bbox(bbox, x, y, w, h):
	cx = x + w * 0.5
	cy = y + h * 0.5
	x = cx - bbox[2] * 0.5
	y = cy - bbox[3] * 0.5
	return [x, y, bbox[2], bbox[3]]


pts_model_ = None
pts_img_size = None

def pts_init(parser):
	ops = parser.parse_args()
	# load configs for gesture
	unparsed = vars(ops)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
	for key in unparsed.keys():
		print('{} : {}'.format(key, unparsed[key]))

	# ---------------------------------------------------------------- 构建模型
	print('Model to use for Hand Pose : %s' % (ops.model))
	
	if ops.model == 'resnet_50':
		model_ = resnet50(num_classes=ops.num_classes, img_size=ops.img_size)
	elif ops.model == 'resnet_18':
		model_ = resnet18(num_classes=ops.num_classes, img_size=ops.img_size)
	elif ops.model == 'resnet_34':
		model_ = resnet34(num_classes=ops.num_classes, img_size=ops.img_size)
	elif ops.model == 'resnet_101':
		model_ = resnet101(num_classes=ops.num_classes, img_size=ops.img_size)
	elif ops.model == "squeezenet1_0":
		model_ = squeezenet1_0(num_classes=ops.num_classes)
	elif ops.model == "squeezenet1_1":
		model_ = squeezenet1_1(num_classes=ops.num_classes)
	elif ops.model == "shufflenetv2":
		model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes)
	elif ops.model == "shufflenet_v2_x1_5":
		model_ = shufflenet_v2_x1_5(pretrained=False, num_classes=ops.num_classes)
	elif ops.model == "shufflenet_v2_x1_0":
		model_ = shufflenet_v2_x1_0(pretrained=False, num_classes=ops.num_classes)
	elif ops.model == "shufflenet_v2_x2_0":
		model_ = shufflenet_v2_x2_0(pretrained=False, num_classes=ops.num_classes)
	elif ops.model == "shufflenet":
		model_ = ShuffleNet(num_blocks=[2, 4, 2], num_classes=ops.num_classes, groups=3)
	elif ops.model == "mobilenetv2":
		model_ = MobileNetV2(num_classes=ops.num_classes)
	elif ops.model == "ReXNetV1":
		model_ = ReXNetV1(num_classes=ops.num_classes)
	elif ops.model == "resnet12":
		model_ = ResNet12(num_classes=ops.num_classes, img_size=ops.img_size)
	elif ops.model == "minivgg":
		model_ = MiniVGG(num_classes=ops.num_classes, img_size=ops.img_size)
	
	use_cuda = torch.cuda.is_available()

	# 加载测试模型
	if os.access(ops.model_path, os.F_OK):  # checkpoint
		#chkpt = torch.load(ops.model_path, map_location='cpu')  # device)
		chkpt=torch.load(ops.model_path, map_location='cuda:0')
		model_.load_state_dict(chkpt)
		print('load test model : {}'.format(ops.model_path))
	device = torch.device("cuda:0" if use_cuda else "cpu")
	model_ = model_.to(device)
	model_.eval()  # 设置为前向推断模式
	global pts_model_
	pts_model_ = model_
	global pts_img_size
	pts_img_size = ops.img_size

def pred_pts(box_lst, img_handle):
	global pts_img_size
	all_pts = []
	for box_idx in range(len(box_lst)):
		bbox = box_lst[box_idx]
		if bbox is None: continue
		img_, x_start, y_start = pad_img(img_handle, bbox[0], bbox[1], bbox[2], bbox[3], 1.5)
		# cv2.imwrite(os.path.join('./test_p21/cropped/', file), img_)
		img_width = img_.shape[1]
		img_height = img_.shape[0]
		# 输入图片预处理
		img_ = cv2.resize(img_, (pts_img_size, pts_img_size), interpolation=cv2.INTER_LINEAR)
		img_ = img_.astype(np.float32)
		img_ = img_ * 1.0 / 255
		img_ = img_.transpose(2, 0, 1)
		img_ = torch.from_numpy(img_).cuda()
		img_ = img_.unsqueeze_(0)
		pre_ = pts_model_(img_.float())  # 模型推理
		output = pre_.cpu().detach().numpy()
		output = np.squeeze(output)
		hand_pts = []
		for i in range(int(output.shape[0] / 2)):
			x = (output[i * 2 + 0] * float(img_width)) + x_start
			y = (output[i * 2 + 1] * float(img_height)) + y_start
			hand_pts.append(x)
			hand_pts.append(y)
		all_pts.append(hand_pts)
	return all_pts
	
def draw_pts(img_handle, all_pts):
	for hand_pts in all_pts:
		pts_hand = {}  # 构建关键点连线可视化结构
		xmin, ymin, xmax, ymax = 9999, 9999, 0, 0
		for i in range(int(len(hand_pts) / 2)):
			x = hand_pts[i * 2 + 0]
			y = hand_pts[i * 2 + 1]
			xmin = min(xmin, x)
			ymin = min(ymin, y)
			xmax = max(xmax, x)
			ymax = max(ymax, y)
			pts_hand[str(i)] = {}
			pts_hand[str(i)] = {
				"x": x,
				"y": y,
			}
		draw_bd_handpose(img_handle, pts_hand, 0, 0)  # 绘制关键点连线
		# ------------- 绘制关键点
		for i in range(int(len(hand_pts) / 2)):
			x = hand_pts[i * 2 + 0]
			y = hand_pts[i * 2 + 1]
			cv2.circle(img_handle, (int(x), int(y)), 3, (255, 50, 60), -1)
			cv2.circle(img_handle, (int(x), int(y)), 1, (255, 150, 180), -1)
	pass
