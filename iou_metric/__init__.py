import numpy as np
import collections
import cv2
 
#定义字典存放颜色分量上下限
#例如：{颜色: [min分量, max分量]}
#{'red': [array([160,  43,  46]), array([179, 255, 255])]}
 
def getColorList():
    dict = collections.defaultdict(list)
    
    # road
    lower_white = np.array([0, 0, 250])
    upper_white = np.array([0, 0, 255])
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    dict['road'] = [(lower_white, upper_white), (lower_orange, upper_orange)]
    
    # water
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    dict['water'] = [(lower_blue, upper_blue)]
 
    return dict

def get_masks(filename):
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color_dict = getColorList()
    
    masks = {}
    for d in color_dict:
        colors = color_dict[d]
        mask = cv2.inRange(hsv, colors[0][0], colors[0][1])
        for color in colors[1:]:
            mask += cv2.inRange(hsv, color[0], color[1])
        masks[d] = mask
    
    return masks

def cal_ious(name0, name1):
    color_dict = getColorList()
    masks0 = get_masks(name0)
    masks1 = get_masks(name1)
    ious = {}
    for d in color_dict:
        mask0 = masks0[d]/255
        mask1 = masks1[d]/255
        
        area0 = mask0.sum()
        area1 = mask1.sum()
        inter = sum(mask0[mask0 == mask1])
        iou = inter / (area0 + area1 - inter)
        
        ious[d] = iou
        
    return ious