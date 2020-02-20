from __future__ import division
 
import torch 
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils.nms import nms 
from utils.parse_config import parse_cfg

import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()

def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("TRT file not found")


class PrepocessYOLO():
    def __init__(self,anchors,conf_thres,nms_thres):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
    
    def process(self,output):
        
        with torch.no_grad():
            for detections in output:
                detections = detections[detections[:,4]>self.conf_thres]
                box_conrner = torch.zeros((detections.shape[0],4),device=detections.device)
                xy = detections[:,0:2]
                wh = detections[:,2:4]/2
                box_conrner[:,0:2] = xy-wh
                box_conrner[:,2:4] = xy+wh
                probabilities = detections[:,4]
                clss_prob,clss = torch.max(detections[:,5:],1)
                nms_indices = nms(box_conrner,probabilities,self.nms_thres)
                main_box_corner = box_corner[nms_indices]
                main_clss = clss[nms_indices]
                main_clss_prob = clss_prob[nms_indices]
                if nms_indices.shape[0] == 0:
                    continue
            
                



class TrtYOLO():
    def __init__(self,engine_file,cfg_file):
        self.engine_file = engine_file
        self.cfg_file = cfg_file
        self.moduel_def = parse_cfg(cfg_file)
        self.net_info = self.moduel_def[0]
        self.inp_w = self.net_info['width']
        self.inp_h = self.net_info['height']
        self.inp_dim = self.inp_h
        self.num_classes = self.net_info['num_classes']
        self.output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)] #yolo3-608
        ##########prep anchors
        #TODO
        ###########
        self.engine = get_engine(self.engine_file)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

    def detection(self,)