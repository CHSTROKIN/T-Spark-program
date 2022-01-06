import consts
import numpy as np
mState = {"WAIT_TO_START" : 0, "WAIT_FOR_GESTURE" : 1, "FAILED" : 2, "GESTURE_OK" : 3, "PASSED" : 4, "WAIT_2" : 5, "WAIT_3" : 6}

cur_text = ""
cur_state = mState["WAIT_TO_START"]
gesture_id = 0
gesture_class = 0
time_start = 0
time_limit = 0
cur_color = (0, 0, 255)
last_update_t = 0
gesture_list = []
cur_pic = None
failed = False
focus_record = np.zeros((consts.point_num, 21), dtype = np.float64)