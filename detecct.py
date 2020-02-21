import argparse
import os
from os.path import isfile, join
import random
import tempfile
import time
import copy
import multiprocessing
import subprocess
import shutil
import cv2


import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from PIL import Image, ImageDraw
import torchvision

from model import trtYOLO
from utils.utils import draw_bboxes,calculate_padding




def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

LABEL_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coco_labels.txt')
ALL_CATEGORIES = load_label_categories(LABEL_FILE_PATH)

def main(target_path,engine_file,cfg_file,cam_device = 0,vanilla_anchor = True,mode = 'image'):
   
    trt_model = trtYOLO(cfg_file,engine_file,vanilla_anchor=True)

    if target_path == None:
        mode = 'video'
    
    print("detect mode is:",mode)
    if mode == 'image':
        detect_single_img(target_path,trt_model)
    elif mode == 'video':
        detect_video(trt_model,cam_device)
    else:
        print("target path error")

def detect_single_img(target_path,trt_model):
    img = cv2.imread(target_path)
    (boxes,clss,clss_prob) = trt_model.detect_frame(img)
    for box,clss_i,cls_prob in zip(boxes,clss,clss_prob):
       print(ALL_CATEGORIES[clss_i]," conf:",cls_prob,box)
    img_with_boxes = draw_bboxes(img,boxes,clss,clss_prob,ALL_CATEGORIES)
    cv2.imwrite(target_path.split('/')[-1],img_with_boxes)


def detect_video(trt_model,cam_device = 0):
    cap = cv2.VideoCapture(cam_device)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    while True:
        print("-----------------------------")
        t_time1 = time.time()
        return_value, frame = cap.read()  
        if return_value is False:
            print("[error]open camera fail !")
            exit()
        t_time2 = time.time()
        print("[Time]get frame time :",(t_time2-t_time1))

        # trt_model.detect_path('data/dog.jpg')
        (boxes,clss,clss_prob) = trt_model.detect_frame(frame)
        if boxes is None:
            img_with_boxes = frame
        else:
            for box,clss_i,cls_prob in zip(boxes,clss,clss_prob):
                print(ALL_CATEGORIES[clss_i]," conf:",cls_prob,box)
            img_with_boxes = draw_bboxes(frame,boxes,clss,clss_prob,ALL_CATEGORIES)
        cv2.imshow("img_with_boxes",img_with_boxes)
        t_time3 = time.time()
        print("[Time]detect time:",(t_time3-t_time2))
        if cv2.waitKey(1) & 0xFF == ord('z'):       # 按q退出
            break
       
        print("[Time]total time: ",(t_time3-t_time1))
        #cv2.imshow("frame",frame)
        print("FPS: ",1./(t_time3- t_time1))
    
    cap.release() 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--cfg_path', type=str, default='model_cfg/yolov3-608.cfg')
    parser.add_argument('--target_path', type=str,default='data/dog.jpg', help='path to target image/video')
    parser.add_argument('--engine_file',type=str,default='yolov3-608.trt',help='path to tensortRT engine file')
    parser.add_argument('--mode',type=str,default='image',help='detect: image or video')
    parser.add_argument('--camera_device',type=int,default=0,help='code of camera device')

    add_bool_arg('vanilla_anchor', default=True, help="whether to use vanilla anchor boxes for training")

    opt = parser.parse_args()

    main(cfg_file=opt.cfg_path,
         target_path=opt.target_path,
         engine_file=opt.engine_file,
         cam_device=opt.camera_device,
         mode=opt.mode,
         vanilla_anchor = True)
    