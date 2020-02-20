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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from PIL import Image, ImageDraw
import torchvision

from models import PrepocessYOLO,TrtYOLO
from utils.utils import draw_bboxes




def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

LABEL_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coco_labels.txt')
ALL_CATEGORIES = load_label_categories(LABEL_FILE_PATH)

def main(target_path,output_path,engine_file,cfg_file,vanilla_anchor = True,mode = 'image'):
    trt_model = TrtYOLO(engine_file)
    prepocessor = PrepocessYOLO(cfg_file,vanilla_anchor=True)

    if target_path == None:
        mode = 'video'
    
    print("detect mode is:",mode)
    if mode == 'image ':
        detect_single_img(target_path,prepocessor,trt_model)
    elif mode == 'video':
        detect_video(prepocessor,trt_model)
    else:
        print("target path error")

def detect_single_img(target_path,prepocessor,trt_model):
    #img = cv2.imread(target_path)
    #w, h = img.size
    #img_ = img[:,:,::-1].transpose((2,0,1))

    img = Image.open(target_path).convert('RGB')
    w, h = img.size
    new_width, new_height = prepocessor.img_size()
    pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
    img_ = torchvision.transforms.functional.pad(img_, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
    img_ = torchvision.transforms.functional.resize(img_, (new_height, new_width))
    
    trt_output = trt_model.inference(img_)
    main_box_corner,clss,clss_prob = prepocessor.process(trt_output)
    
    img_with_boxes = Image.open(target_path)
    boxes = np.zeros(shape = main_box_corner.shape)
    
    boxes[:,0] = main_box_corner[:, 0].to('cpu').item() / ratio - pad_w
    boxes[:,1] = main_box_corner[:, 1].to('cpu').item() / ratio - pad_h
    boxes[:,2] = main_box_corner[:, 2].to('cpu').item() / ratio - pad_w
    boxes[:,3] = main_box_corner[:, 3].to('cpu').item() / ratio - pad_h 
    
    img_with_boxes = draw_bboxes(img,boxes,clss_prob,clss,ALL_CATEGORIES)
    img_with_boxes.save(target_path.split('/')[-1])
    


def detect_video(Prepocessor,trt_model,device = 0,save_img = True,output_path = 'output/'):

    cap = cv.VideoCapture(device)  # 打开摄像头
    cap.set(cv.CAP_PROP_FRAME_WIDTH,1024)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)

    while True:
        return_value, img = cap.read() 
        frame = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        if return_value is False:
            print('read video error')
            exit()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--cfg_file', type=str, default='model_cfg/yolov3-608.cfg')
    parser.add_argument('--target_path', type=str,default='data/dog.jpg', help='path to target image/video')
    parser.add_argument('--output_path', type=str, default="outputs/visualization/")
    parser.add_argument('--weights_path', type=str, default='yolov3.weights',help='path to weights file')
    parser.add_argument('--engine_file',type=str,default='yolov3-608.trt',help='path to tensortRT engine file')

    add_bool_arg('vanilla_anchor', default=True, help="whether to use vanilla anchor boxes for training")

    opt = parser.parse_args()

    main(target_path=opt.target_path,
         output_path=opt.output_path,
         engine_file=opt.engine_file,
         cfg_file=opt.cfg_file,
         vanilla_anchor = True)
    