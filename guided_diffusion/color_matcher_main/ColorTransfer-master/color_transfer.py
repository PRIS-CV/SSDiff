import cv2
import numpy as np
import argparse
from config import *

def get_ori_color_num(ori_img_path):
    img = cv2.imread(ori_img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    height, width, _ = img.shape
    # red,orange,yellow,green,cyan,blue,purple
    ori_color_num = [0, 0, 0, 0, 0, 0, 0]
    offset_of_piexls = {}
    for i in range(0, height-1):
        for j in range(0, width-1):
            cur_index = i*width+j
            if h[i][j] >= RED_MIN:
                ori_color_num[0] = ori_color_num[0]+1
                offset_of_piexls[cur_index] = {'index': 0, 'offset': h[i][j]-RED_MIN}
            elif h[i][j] <= RED_MAX:
                ori_color_num[0] = ori_color_num[0]+1
                offset_of_piexls[cur_index] = {'index': 0, 'offset': h[i][j]+18}
            elif ORANGE_MIN <= h[i][j] <= ORANGE_MAX:
                ori_color_num[1] = ori_color_num[1]+1
                offset_of_piexls[cur_index] = {'index': 1, 'offset': h[i][j]-ORANGE_MIN}
            elif YELLOW_MIN <= h[i][j] <= YELLOW_MAX:
                ori_color_num[2] = ori_color_num[2]+1
                offset_of_piexls[cur_index] = {'index': 2, 'offset': h[i][j]-YELLOW_MIN}
            elif GREEN_MIN <= h[i][j] <= GREEN_MAX:
                ori_color_num[3] = ori_color_num[3]+1
                offset_of_piexls[cur_index] = {'index': 3, 'offset': h[i][j]-GREEN_MIN}
            elif CYAN_MIN <= h[i][j] <= CYAN_MAX:
                ori_color_num[4] = ori_color_num[4]+1
                offset_of_piexls[cur_index] = {'index': 4, 'offset': h[i][j]-CYAN_MIN}
            elif BLUE_MIN <= h[i][j] <= BLUE_MAX:
                ori_color_num[5] = ori_color_num[5]+1
                offset_of_piexls[cur_index] = {'index': 5, 'offset': h[i][j]-BLUE_MIN}
            elif PURPLE_MIN <= h[i][j] <= PURPLE_MAX:
                ori_color_num[6] = ori_color_num[6]+1
                offset_of_piexls[cur_index] = {'index': 6, 'offset': h[i][j]-PURPLE_MIN}
    return ori_color_num,offset_of_piexls,height,width,hsv,s,v

def get_ref_color_num(ref_img_path):
    img = cv2.imread(ref_img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    height, width, _ = img.shape
    # red,orange,yellow,green,cyan,blue,purple
    ref_color_num = [0, 0, 0, 0, 0, 0, 0]
    for i in range(0, height-1):
        for j in range(0, width-1):
            if h[i][j] >= RED_MIN or h[i][j] <= RED_MAX:
                ref_color_num[0] = ref_color_num[0]+1
            elif ORANGE_MIN <= h[i][j] <= ORANGE_MAX:
                ref_color_num[1] = ref_color_num[1]+1
            elif YELLOW_MIN <= h[i][j] <= YELLOW_MAX:
                ref_color_num[2] = ref_color_num[2]+1
            elif GREEN_MIN <= h[i][j] <= GREEN_MAX:
                ref_color_num[3] = ref_color_num[3]+1
            elif CYAN_MIN <= h[i][j] <= CYAN_MAX:
                ref_color_num[4] = ref_color_num[4]+1
            elif BLUE_MIN <= h[i][j] <= BLUE_MAX:
                ref_color_num[5] = ref_color_num[5]+1
            elif PURPLE_MIN <= h[i][j] <= PURPLE_MAX:
                ref_color_num[6] = ref_color_num[6]+1
    return ref_color_num


def get_h_value(ori_index, ref_index, offset):
    h_v = 0
    base_dict = {0:NUM_OF_RED,1:NUM_OF_ORANGE,2:NUM_OF_YELLOW,3:NUM_OF_GREEN,4:NUM_OF_CYAN,5:NUM_OF_BLUE,6:NUM_OF_PURPLE}
    base = base_dict[ori_index]

    if ref_index == 0:
        new_offset = offset/base*NUM_OF_RED
        if new_offset < 180-RED_MIN:
            h_v = RED_MIN+new_offset
        elif new_offset >= 180-RED_MIN:
            h_v = new_offset-(180-RED_MIN)
    elif ref_index == 1:
        h_v = (offset/base*NUM_OF_ORANGE)+ORANGE_MIN
    elif ref_index == 2:
        h_v = (offset/base*NUM_OF_YELLOW)+YELLOW_MIN
    elif ref_index == 3:
        h_v = (offset/base*NUM_OF_GREEN)+GREEN_MIN
    elif ref_index == 4:
        h_v = (offset/base*NUM_OF_CYAN)+CYAN_MIN
    elif ref_index == 5:
        h_v = (offset/base*NUM_OF_BLUE)+BLUE_MIN
    elif ref_index == 6:
        h_v = (offset/base*NUM_OF_PURPLE)+PURPLE_MIN
    return h_v

def color_transfer(ori_img_path,ref_img_path,new_img_path):
    ori_color_num, offset_of_piexls,height,width,hsv,s,v = get_ori_color_num(ori_img_path)
    ori_color_arg = list(np.argsort(np.array(ori_color_num)))
    ref_color_num = get_ref_color_num(ref_img_path)
    ref_color_arg = list(np.argsort(np.array(ref_color_num)))

    for i in range(0, height-1):
        for j in range(0, width-1):
            cur_index = i*width+j
            ori_index = offset_of_piexls[cur_index]['index']
            offset = offset_of_piexls[cur_index]['offset']
            ref_index = ref_color_arg[ori_color_arg.index(ori_index)]
            h_v = get_h_value(ori_index, ref_index, offset)
            hsv[i][j] = tuple([h_v, s[i][j], v[i][j]])
    new_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(new_img_path, new_bgr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--origin", required = True,help = "Path to the origin image")
    ap.add_argument("-r", "--reference", required = True,help = "Path to the reference image")
    ap.add_argument("-g", "--generated", required = True,help = "Path to the generated image")
    args = vars(ap.parse_args())

    color_transfer(args["origin"],args["reference"],args["generated"])