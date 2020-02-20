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
vanilla_anchor_list = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
vanilla_anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]

def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("TRT file not found")


class PrepocessYOLO():
    def __init__(self,cfg_file,vanilla_anchor):
        self.cfg_file = cfg_file
        self.moduel_def = parse_cfg(cfg_file)
        self.net_info = self.moduel_def[0]
        self.inp_w = self.net_info['width']
        self.inp_h = self.net_info['height']
        self.inp_dim = self.inp_h
        self.num_classes = self.net_info['num_classes']

        ####### anchors TODO
        if vanilla_anchor:
            self.anchors = vanilla_anchor_list
            self.yolo_masks = vanilla_anchor_mask
        else:
            self.yolo_masks = [[int(y) for y in x.split(',')] for x in net_info["yolo_masks"].split('|')]
            self.anchors  = [[float(y) for y in x.split(',')] for x in row.split("'")[0].split('|')]
        ########################
        self.num_anchors = len(anchors)

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
        self.output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)] #yolo3-608
        
        self.engine = get_engine(self.engine_file)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

    def inference(self,img):
        CUDA = torch.cuda.is_available()
        with torch.no_grad():
            inference_start = time.time()
            self.inputs[0].host = img[0]
            trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
            inference_end = time.time() 
        return trt_outputs


