from hand_track.hand_data_iter.datasets import draw_bd_handpose

import torch
import torch.nn as nn

import os, sys, time, argparse, shutil, importlib, random
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import chinese_display as chn

# 摄像头
import cv2

# 项目路径
proj_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
# 导入识别库路径
sys.path.append(os.path.join(proj_path, 'gesture_recog'))
import gesture_recog  as gr

# 导入检测库路径
det_path = os.path.join(proj_path, 'det_model')
sys.path.append(os.path.join(det_path, 'det_forward'))
sys.path.append(os.path.join(proj_path, 'hand_track'))
import det_pts_bbox
import pred_pts
import states
import consts
import math

# __________________________________________________________________________
all_colors = ["red", "purple", "blue", "white", "green", "yellow"]

purple_mask = [[135, 20, 100], [175, 100, 255]]
yellow_mask = [[8, 80, 140], [20, 130, 180]]  # unused


def get_img_path(color, is_real, pose, category):
    cat = "real" if is_real else "fake"
    root = os.path.join("./", category, cat, f"pose{pose}")
    if not os.path.exists(root):
        os.makedirs(root)
    return f"./{category}/{cat}/pose{pose}/{color}.jpg"


def make_hsv(color, is_real, pose):
    img_path = get_img_path(color, is_real, pose, "img")
    hsv_path = get_img_path(color, is_real, pose, "hsv")
    img = cv2.imread(img_path, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(hsv_path, hsv)


def make_mask(color, is_real, pose, color_mask):
    pass


def make_mask(frame):
    # mask_path = get_img_path(color, is_real, pose, "mask")
    # hsv_path = get_img_path(color, is_real, pose, "hsv")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # change to hsv format file
    dst = cv2.GaussianBlur(hsv, (35, 35), 0)
    # Kernel size: (35,35), must be odd numbers, larger values mean softer edges & less noise
    lower_range = np.array(purple_mask[0])
    upper_range = np.array(purple_mask[1])
    mask = cv2.inRange(dst, lower_range, upper_range)
    # cv2.imwrite(mask_path, mask)
    return mask, hsv


def checkcoords(coordx, coordy, img):
    cnt = 0
    for i in range(-10, 10):
        for j in range(-10, 10):
            if img[i][j].all():
                cnt += 1
    if cnt > 10:
        return 1
    return 0


def checkpic(coords, img):
    # print('coords size:',len(coords))
    msum = 0.0
    for i in range(0, 2):
        msum -= checkcoords(coords[2 * i], coords[2 * i + 1], img) * 2
    for i in range(2, 21):
        msum += checkcoords(coords[2 * i], coords[2 * i + 1], img)
    if (msum < 10):
        return True
    return False


'''
def read_file(f):
    file = open(f)
    return [line.split()[1:] for line in file.readlines()]
'''
# fake_hand=False
rname = ""


def check_is_screen(frame, bones):
    mask, hsv = make_mask(frame)
    global rname
    if checkpic(bones, mask):
        print('real hand')
    # if random.random() > 0.9:
    # rname = str(random.randint(0, 10000))
    # rdpath_hsv = f"{rname}+real_hsv.jpg"
    # rdpath_mask = f"{rname}+real_mask.jpg"
    # cv2.imwrite(rdpath_hsv, hsv)
    # cv2.imwrite(rdpath_mask, mask)
    # return True
    elif consts.purple:
        print('fake hand')
        states.failed = True


# if random.random() > 0.9:
# rname = str(random.randint(0, 10000))
# rdpath_hsv = f"{rname}+fake_hsv.jpg"
# rdpath_mask = f"{rname}+fake_mask.jpg"
# cv2.imwrite(rdpath_hsv, hsv)
# cv2.imwrite(rdpath_mask, mask)

# fake_hand=True
# time.sleep(1)
# return False

# __________________________________________________________________________
def read_classes_ok():
    classes_ok = {}
    lines = open("classes_ok.txt").readlines()
    for l in lines:
        sp = l.strip().split(' ')
        classes_ok[int(sp[1])] = sp[0]
    return classes_ok


'''
class tracker :
    __ = 0
    def __init__(self, freq):
        self.frame_ind = 0
		self.crops = []
		self.box_lst = []
		self.labels = []

    def track(self, frame, freq_recog):
		self.frame_ind += 1
		self.frame = cv2.flip(self.frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
		# img = frame
		self.crops, self.box_lst = det_pts_bbox.det_process(frame)  # 检测
		all_pts = pred_pts.pred_pts(self.box_lst, frame)
		self.labels = gr.rec_hands(self.crops)
		# print(labels)
		# print(all_pts)

		###画图
		pred_pts.draw_pts(frame, all_pts)
		for idx, bbox in enumerate(self.box_lst):
			frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0))
			cv2.putText(frame, str(self.labels[idx]), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(0, 0, 255), 2)
		cv2.imshow("video", frame)
'''


def area(box):
    if box is None:
        return 0
    return box[2] * box[3]


def intersection(box1, box2):
    t = (max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[0] + box1[2], box2[0] + box2[2]),
         min(box1[1] + box1[3], box2[1] + box2[3]))
    if (t[0] > t[2] or t[1] > t[3]):
        return None
    return (t[0], t[1], t[2] - t[0], t[3] - t[1])


def feature_check(frame_1, frame_2):
    ls = []
    ls.append(abs(frame_1[4] - frame_2[4]))
    ls.append(abs(frame_1[8] - frame_2[8]))
    ls.append(abs(frame_1[12] - frame_2[12]))
    ls.append(abs(frame_1[16] - frame_2[16]))
    ls.append(abs(frame_1[20] - frame_2[20]))
    ls.sort(reverse=True)
    return (ls[0] + ls[1] + ls[2]) / (3.0)


def check_is_continued1(frame_1, frame_2, frame_3):
    test_continue = frame_2[0] - frame_1[0]
    test_continue_2 = frame_3[0] - frame_2[0]
    test_continue = test_continue * test_continue
    test_continue_2 = test_continue_2 * test_continue_2
    length = math.sqrt(test_continue.sum()) / (frame_2[1] - frame_1[1])
    length_2 = math.sqrt(test_continue_2.sum()) / (frame_3[1] - frame_2[1])
    length = abs(length_2 - length)
    return length


def check_is_continued2(frame_1, frame_2, frame_3):
    test_continue = (frame_2[0] - frame_1[0]) / (frame_2[1] - frame_1[1])
    test_continue_2 = (frame_3[0] - frame_2[0]) / (frame_3[1] - frame_2[1])
    dif = []
    for idx in range(21):
        dx = test_continue_2[2 * idx + 0] - test_continue[2 * idx + 0]
        dy = test_continue_2[2 * idx + 1] - test_continue[2 * idx + 1]
        d = math.sqrt(dx * dx + dy * dy)
        dif.append(d)
    dif.sort(reverse=True)
    ret = 0.0
    k = 5
    for i in range(k):
        ret += dif[i]
    ret /= k
    return ret


def _debug(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print("click %.2f %.2f" % (x, y))


def check_is_continued(frame_1, frame_2, frame_3):
    return check_is_continued1(frame_1, frame_2, frame_3)


def check_is_continued_key(frame_1, frame_2, frame_3):
    length = feature_check(frame_1[0], frame_2[0])
    length_2 = feature_check(frame_2[0], frame_3[0])
    length = abs(length_2 - length)
    return length


def Laplacian(img):
    t = cv2.Laplacian(img, cv2.CV_64F)
    ans = t.var()
    return ans


def add_purple_background(frame):
    W = int(1.8 * 480)
    H = int(1.8 * 640)
    ret = np.full(shape=(W, H, 3), fill_value=(200, 0, 200), dtype=np.uint8)
    ret[int(0.6 * 480):int(0.6 * 480) + frame.shape[0], int(640 * 0.6):int(640 * 0.6) + frame.shape[1], :] = frame
    if (not consts.purple) :
        return frame
    return ret


def hand_track_realtime(parser, freq=5, freq_label=5, width=480, height=640, interrupt_when_lost=False):
    # print('parser', parser, dir(parser))
    # exit()
    capture = cv2.VideoCapture(0)  # 0为电脑内置摄像头

    cv2.waitKey(3000)
    capture.set(3, width)
    capture.set(4, height)

    ###全局初始化###
    det_pts_bbox.detrac_lobal_init(parser)
    pred_pts.pts_init(parser)
    gr.rec_init(parser)

    classes_ok = read_classes_ok()
    classes = gr.rec_data[2]

    ###开始循环读取摄像头
    start_time = time.time()
    frame_ind = -1
    frame_ind_for_label = 0
    labels = []
    lst_pts = []
    _max = 1.0
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    # f = 1
    while (True):
        if states.cur_state == states.mState["WAIT_FOR_GESTURE"]:
            states.gesture_class = states.gesture_list[states.gesture_id]
        # capture.set(cv2.CAP_PROP_FOCUS, f);

        # f += 1
        # print(capture.get(cv2.CAP_PROP_FOCUS), f)
        ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        if ret is False:
            print('capture open failed!')
            break

        frame_ind += 1
        frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
        # img = frame
        # crops, box_lst = det_pts_bbox.det_process(frame)  # 检测
        _crops, _box_lst = det_pts_bbox.det_process(frame)  # 检测
        crops = []
        box_lst = []
        if len(_box_lst) > 0:
            idx_lst = [0]
            for i in range(1, len(_box_lst)):
                b = _box_lst[i]
                another = _box_lst[idx_lst[0]]
                if (area(intersection(b, another)) * 1.0 / min(area(b), area(another)) > 0.7):
                    if (area(b) > area(another)):
                        idx_lst[0] = i
                else:
                    idx_lst.append(i)
            for i in idx_lst:
                crops.append(_crops[i])
                box_lst.append(_box_lst[i])
        if (len(_box_lst) == 0 and interrupt_when_lost):
            states.failed = True

        all_pts = pred_pts.pred_pts(box_lst, frame)
        if (len(box_lst) > 0):
            lst_pts.append((np.array(all_pts[0], dtype=np.float64), time.time()))
        else:
            lst_pts = []

        if (len(lst_pts) >= 1):
            # print('it is bigger then one', len(lst_pts[0][0]))
            check_is_screen(frame, lst_pts[0][0])

        if len(lst_pts) > 3:
            lst_pts = lst_pts[len(lst_pts) - 3: len(lst_pts)]

        if len(lst_pts) >= 3:
            length = check_is_continued(lst_pts[0], lst_pts[1], lst_pts[2])
            length /= math.sqrt(1.0 * area(box_lst[0]))
            _max = max(_max, length)
            if (length > 30):
                states.failed = True
                print('你是PPT ')
            cv2.putText(frame, str(_max), (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        if (frame_ind_for_label % freq_label == 0):
            labels = gr.rec_hands(crops)
            _max = 0

        # print(labels)
        # print(all_pts)

        ###画图
        pred_pts.draw_pts(frame, all_pts)
        for idx, bbox in enumerate(box_lst):
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0))
            if (frame_ind_for_label % freq_label == 0):
                cv2.putText(frame, str(labels[idx]), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                print(labels[idx] + " detected")
            else:
                pass
        # cv2.putText(frame, "not detecting", (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 临时策略

        if len(box_lst) != 1:
            states.failed = True

        if states.cur_text != "":
            frame = chn.showText(frame, position=(50, 430), s=states.cur_text, size=20, color=states.cur_color)
        if states.cur_state == states.mState["WAIT_FOR_GESTURE"]:
            t = time.time()
            if not (states.cur_pic is None):
                if t - states.last_update_t > 100:
                    bar = np.zeros((250, 640 - 250, 3))
                    k = (t - states.time_start) / states.time_limit
                    if k <= 1:
                        _ = int(k * 390)
                        bar[100:150, :_, 1] = np.full((50, _), 255 * (1 - k) + k * 0)
                        bar[100:150, :_, 2] = np.full((50, _), 0 * (1 - k) + k * 255)
                        states.cur_pic = np.hstack([img, Image.fromarray(np.uint8(bar))])
                frame = np.vstack([frame, states.cur_pic])
            else:
                img = Image.open(os.path.join(r".\pic", classes[states.gesture_class] + ".jpg"))
                img = img.resize((250, 250))
                states.cur_pic = np.hstack([img, Image.fromarray(np.zeros((250, 640 - 250, 3)), 'RGB')])
                frame = np.vstack([frame, states.cur_pic])
        if states.cur_state == states.mState["WAIT_2"] or states.cur_state == states.mState["WAIT_3"]:
            states.wait2_ok = True
            for idx, point in enumerate(consts.fingers_pos):
                if len(box_lst) == 1 and len(all_pts[0]) > 0:
                    ind = consts.fingers_idx[idx]
                    correct_p = np.array(point)
                    # print(len(all_pts[0]))
                    cur_p = np.array([all_pts[0][ind * 2 + 0], all_pts[0][ind * 2 + 1]])
                    dis = math.sqrt(np.sum((correct_p - cur_p) ** 2))
                else:
                    dis = 1e9
                mdis = consts.max_dis
                if (states.cur_state == states.mState["WAIT_3"]):
                    mdis *= 2
                if dis <= mdis:
                    if states.cur_state == states.mState["WAIT_2"]:
                        cv2.circle(frame, (int(point[0]), int(point[1])), consts.max_dis, (0, 255, 0), 2)
                    else:
                        cv2.circle(frame, (int(point[0]), int(point[1])), consts.max_dis, (0, 255, 255), 2)
                else:
                    cv2.circle(frame, (int(point[0]), int(point[1])), consts.max_dis, (0, 0, 255), 2)
                    if states.cur_state == states.mState["WAIT_2"]:
                        states.wait2_ok = False
                    else:
                        states.failed = True
            if not (len(labels) == 1 and "five" in labels):
                if states.cur_state == states.mState["WAIT_2"]:
                    states.wait2_ok = False
                else:
                    states.failed = True

        if frame_ind % 5 == 0:
            end_time = time.time()
            print('fps: %d' % (int(5 / (end_time - start_time))))  # 程序运行fps
            start_time = end_time

       # frame = add_purple_background(frame)

        cv2.imshow("video", frame)
        if "_first" not in vars():
            cv2.setMouseCallback("video", _debug)
            _first = False
        c = cv2.waitKey(1)

        frame_ind_for_label += 1
        if c == 27:  # esc
            capture.release()
            cv2.destroyAllWindows()
            break
        if states.cur_state == states.mState["WAIT_TO_START"]:
            states.cur_color = (0, 0, 0)
            states.cur_text = "按下 Space 键开始验证!"
            if c == 32:
                frame_ind_for_label = 0
                labels = []
                states.failed = False
                states.cur_state = states.mState["WAIT_2"]
                states.cur_color = (255, 0, 0)
                states.cur_text = "请将右手置于镜头前，移至合适距离并摆正，使指尖与圆圈重合"
        elif states.cur_state == states.mState["WAIT_2"]:
            if states.failed:
                states.cur_pic = None
                states.cur_state = states.mState["FAILED"]
                states.cur_color = (255, 0, 0)
                states.cur_text = "验证失败!"
                states.time_failed = time.time()
            elif states.wait2_ok:
                capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                states.cur_focus = int(10)
                states.cur_state = states.mState["WAIT_3"]
                states.cur_color = (0, 255, 255)
                states.cur_text = "检测中，请不要移动！"
        elif states.cur_state == states.mState["WAIT_3"]:
            if states.failed:
                states.cur_pic = None
                states.cur_state = states.mState["FAILED"]
                states.cur_color = (255, 0, 0)
                states.cur_text = "验证失败!"
                states.time_failed = time.time()
            elif states.cur_focus < 200:
                capture.set(cv2.CAP_PROP_FOCUS, states.cur_focus)
                _ret, _frame = capture.read()
                _frame = cv2.flip(_frame, 1)
                for i in range(len(consts.focus_ranges)):
                    t = np.array(_frame)[consts.point1[i][1]:consts.point2[i][1],
                        consts.point1[i][0]:consts.point2[i][0], :]
                    v = Laplacian(t)
                    states.focus_record[i][int(states.cur_focus / 10)] = v
                # cv2.imshow("qwq", t)
                states.cur_focus += 10
            else:
                capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                cv2.waitKey(100)
                for i in range(len(consts.focus_ranges)):
                    best_f = np.argmax(states.focus_record[i])
                    print(i, best_f * 10)
                    print(states.focus_record[i])
                    if not (consts.focus_ranges[i][1] >= best_f * 10 >= consts.focus_ranges[i][0]):
                        states.failed = True
                        states.cur_pic = None
                        states.cur_state = states.mState["FAILED"]
                        states.cur_color = (255, 0, 0)
                        states.cur_text = "验证失败!"
                        states.time_failed = time.time()
                        break

                if not states.failed:
                    frame_ind_for_label = 0
                    labels = []
                    states.cur_state = states.mState["WAIT_FOR_GESTURE"]
                    states.gesture_id = 0
                    states.gesture_list = random.sample(list(classes_ok.keys()), 3)
                    random.shuffle(states.gesture_list)
                    # print(gesture_class)
                    states.time_start = time.time()
                    states.time_limit = consts.time_limit
        elif states.cur_state == states.mState["WAIT_FOR_GESTURE"]:
            lef = states.time_start + states.time_limit - time.time()
            # 待修改
            if not states.failed and classes[states.gesture_class] in labels:
                states.cur_pic = None
                if states.gesture_id + 1 < consts.GESTURE_NUM:
                    states.cur_state = states.mState["GESTURE_OK"]
                    states.cur_color = (0, 255, 0)
                    states.cur_text = "Gesture %s OK" % classes[states.gesture_class]
                    states.time_ok = time.time()
                else:
                    states.cur_state = states.mState["PASSED"]
                    states.cur_color = (0, 255, 0)
                    states.cur_text = "验证通过!"
                    states.time_passed = time.time()
            elif lef < 0 or states.failed:
                states.cur_pic = None
                states.cur_state = states.mState["FAILED"]
                states.cur_color = (255, 0, 0)
                states.cur_text = "验证失败!"
                states.time_failed = time.time()
            else:
                states.cur_color = (255, 0, 0)
                states.cur_text = "请缓慢做出手势 %d / %d : %s 剩余 %.1f 秒!" % (
                    states.gesture_id + 1, consts.GESTURE_NUM, classes[states.gesture_class], lef)
        elif states.cur_state == states.mState["FAILED"]:
            if time.time() - states.time_failed > 4:
                states.cur_state = states.mState["WAIT_TO_START"]
        elif states.cur_state == states.mState["GESTURE_OK"]:
            if time.time() - states.time_ok > 0.5:
                frame_ind_for_label = 0
                states.cur_state = states.mState["WAIT_FOR_GESTURE"]
                labels = []
                states.gesture_id += 1
                # print(gesture_class)
                states.time_start = time.time()
                states.time_limit = consts.time_limit
        elif states.cur_state == states.mState["PASSED"]:
            if time.time() - states.time_passed > 4:
                states.cur_state = states.mState["WAIT_TO_START"]
    capture.release()
    cv2.destroyAllWindows()
