from argparse import ArgumentParser
import yaml
import sys
import os
import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import mediapipe as mp

detrac_type = 'mediapipe'
#detrac_type = 'pytorch'
models = []

def cal_envolop_box(p21):
    xmin, ymin, xmax, ymax = 9999, 9999, 0, 0
    for i in range(21):
        xmin = min(xmin, p21[i*2])
        ymin = min(ymin, p21[i*2+1])
        xmax = max(xmax, p21[i*2])
        ymax = max(ymax, p21[i*2+1])
    return [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]
    #return [xmin, ymin, xmax-xmin, ymax-ymin]

def detrac_lobal_init(parser):
	args = parser.parse_args()
	global models, detrac_type
	if detrac_type == 'mediapipe':
		mp_hands = mp.solutions.hands
		model = mp_hands.Hands(min_detection_confidence=args.detection_thresh, min_tracking_confidence=args.tracking_thresh)
		models = [model]
	elif detrac_type == 'pytorch':
		pass
	else:
		pass
	pass

def detrac_predict(frame):
	image_width = frame.shape[1]
	image_height = frame.shape[0]
	global models, detrac_type
	if detrac_type == 'mediapipe':
		mp_hands = mp.solutions.hands
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image.flags.writeable = False
		results = models[0].process(image)
		all_hands = []
		if results.multi_hand_landmarks:
			for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
				hand_pts = []
				#print(results.multi_handedness[hand_idx])
				for idx, mark in enumerate(mp_hands.HandLandmark):
					#print(idx, mark)
					hand_pts.append(hand_landmarks.landmark[mark].x * image_width)
					hand_pts.append(hand_landmarks.landmark[mark].y * image_height)
				all_hands.append(hand_pts)
		return all_hands
		pass
	elif detrac_type == 'pytorch':
		pass
	else:
		pass
	pass
	
def draw_pts(img_handle, all_hands):
	draw_bd_handpose(img_handle, pts_hand, 0, 0)  # 绘制关键点连线
	# ------------- 绘制关键点
	for i in range(int(output.shape[0] / 2)):
		x = (output[i * 2 + 0] * float(img_width)) + x_start
		y = (output[i * 2 + 1] * float(img_height)) + y_start
		cv2.circle(img_handle, (int(x), int(y)), 3, (255, 50, 60), -1)
		cv2.circle(img_handle, (int(x), int(y)), 1, (255, 150, 180), -1)
	pass




def preprocess(img, net_input_size):
    sz = img.shape
    h = sz[0]
    w = sz[1]
    #print(sz)
    if h < w:
        data = np.zeros((w, w, 3))
        data[:,:,0] = 123
        data[:,:,1] = 117
        data[:,:,2] = 104
        data[(w - h) // 2:h + (w - h) // 2, :, :] = img
    else:
        data = np.zeros((h, h, 3))
        data[:, :, 0] = 123
        data[:, :, 1] = 117
        data[:, :, 2] = 104
        data[:, (h - w) // 2:w + (h - w) // 2, :] = img
    data = cv2.resize(data, (net_input_size[0], net_input_size[1]))
    data = data[np.newaxis, :, :, :]
    data = np.array(data, dtype=np.float32)
    data = data / 255
    data = torch.from_numpy(data).permute(0, 3, 1, 2)
    #print(data.shape)

    return data

def __index_str(name: str, index: int):
    return name + '_{}'.format(index)

def decode(coord: list, conf: list, cls: list, net_input_size_: [int, int], reductions, num_heads: int, anchors):
    assert len(coord) == len(conf)
    assert len(coord) == len(cls)
    assert len(coord) == len(anchors)

    out_coord = []
    out_score = []
    out_obj_score = []
    for i in range(num_heads):
        cur_coord = coord[i].clone()
        cur_conf = conf[i].clone()
        cur_cls = cls[i].clone()
        device = cur_coord.device
        cur_anchors_wh = anchors[i][:, 2:].to(device)
        nW = net_input_size_[0] // reductions[i]
        nH = net_input_size_[1] // reductions[i]
        nA = cur_anchors_wh.size(0)
        nB = cur_coord.size(0)
        nC = cur_cls.size(-1)

        pos_x = torch.arange(0, nW, 1, dtype=torch.float32, device=device)
        pos_y = torch.arange(0, nH, 1, dtype=torch.float32, device=device)
        cur_coord[:, :, :, :, 0] = (cur_coord[:, :, :, :, 0] + pos_x.view(1, 1, 1, nW)) \
            * reductions[i]
        cur_coord[:, :, :, :, 1] = (cur_coord[:, :, :, :, 1] + pos_y.view(1, 1, nH, 1)) \
            * reductions[i]

        cur_coord[:, :, :, :, 2:] = cur_coord[:, :, :, :, 2:].exp() \
            * cur_anchors_wh.view(1, nA, 1, 1, 2)
        out_coord.append(cur_coord.view(nB, -1, 4))
        score = cur_conf * cur_cls
        out_score.append(score.view(nB, -1, nC))
        out_obj_score.append(cur_conf.view(nB, -1))

    out_coord = torch.cat(out_coord, dim=1)
    out_score = torch.cat(out_score, dim=1)
    out_obj_score = torch.cat(out_obj_score, dim=1)
    out_coord[:, :, :2] = out_coord[:, :, :2] - out_coord[:, :, 2:] / 2
    return out_coord, out_score, out_obj_score

def postprocess(pred, cls_loss_type, conf_thresh, num_class, nms_param, anchors, reductions, net_input_size_, num_heads):
    softmax = nn.Softmax(dim=-1)


    loc_pred = []
    conf_pred = []
    cls_pred = []
    for i in range(num_heads):
        loc_i = pred[__index_str('loc', i)].detach()
        loc_i[:, :, :, :, 0:2] = loc_i[:, :, :, :, 0:2].sigmoid()
        loc_pred.append(loc_i)
        conf_i = pred[__index_str('conf', i)].detach().sigmoid()
        conf_pred.append(conf_i)
        if cls_loss_type == 'ce':
            cls_i = softmax(pred[__index_str('cls', i)].detach())
        else:
            cls_i = pred[__index_str('cls', i)].detach().sigmoid()
        cls_pred.append(cls_i)
    loc_pred_xywh, all_cls_conf_pred, obj_conf_pred = decode(loc_pred, conf_pred, cls_pred, net_input_size_, reductions, num_heads, anchors)
    obj_conf_mask = obj_conf_pred > conf_thresh

    result = []
    for class_id in range(num_class):
        class_conf_pred = all_cls_conf_pred[0, :, class_id]
        c_mask = (class_conf_pred > conf_thresh) & obj_conf_mask[0]
        scores = class_conf_pred[c_mask]
        if scores.size(0) == 0:
            result.append(None)
            continue
        l_mask = c_mask.view(-1, 1).expand_as(loc_pred_xywh[0, :, :])
        boxes = loc_pred_xywh[0, :, :][l_mask].contiguous().view(-1, 4)
        res_boxes, res_scores = nms(boxes, scores, **nms_param)
        #res_boxes = self.reverser.reverse_box(res_boxes)
        result.append((res_boxes, res_scores))
    return result


def load_config(path):
	with open(path, 'r') as fin:
		config = yaml.load(fin)
	return config


def det_init():
    detrac_lobal_init()
    pass

def expand_bbox(box, scale=1.0):
    cx = box[0] + box[2]*0.5
    cy = box[1] + box[3]*0.5
    w = box[2]*scale
    h = box[3]*scale
    box = [cx-w*0.5, cy-w*0.5, w, h]
    return box

def expand_and_pad(im, box, scale=1.0):
    cx = box[0] + box[2]*0.5
    cy = box[1] + box[3]*0.5
    boxLen = max(box[2], box[3])
    invXOffset = cx - boxLen*scale*0.5
    invYOffset = cy - boxLen*scale*0.5
    M = [[1.0, 0, -invXOffset],[0, 1.0, -invYOffset]]
    M = np.array(M)
    im = cv2.warpAffine(im, M, (int(boxLen*scale), int(boxLen*scale)), flags=cv2.INTER_LINEAR)
    return im

# Specifically for T-Spark
def det_process(img):
    all_hands = detrac_predict(img)
    all_bboxes = []
    for i in range(len(all_hands)):
        bbox = cal_envolop_box(all_hands[i])
        all_bboxes.append(bbox)
    crops = []
    # xy_start = []
    box_lst = []
    for bbox in all_bboxes:
        crop = expand_and_pad(img, bbox, scale=1.5)
        crops.append(crop)
        box_lst.append(bbox)
    return crops, box_lst
