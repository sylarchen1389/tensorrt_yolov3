from __future__ import division
 
import torch 
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils.nms import nms 
from utils.parse_config import parse_cfg
from utils.utils import predict_transform

import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


TRT_LOGGER = trt.Logger()
# vanilla_anchor_list = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
vanilla_anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]
vanilla_anchor_list =  [[(116, 90), (156, 198), (373, 326)],
                             [(30, 61),  (62, 45),   (59, 119)],
                             [(10, 13),  (16, 30),   (33, 23)]]


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
        self.inp_w = int(self.net_info['width'])
        self.inp_h = int(self.net_info['height'])
        self.nms_thres = float(self.net_info['nms_thres'])
        self.conf_thres = float(self.net_info['conf_thres'])
        self.inp_dim = self.inp_h
        self.num_classes =int( self.net_info['num_classes'])
        self.output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)] #yolo3-608
        ####### anchors TODO
        if vanilla_anchor:
            self.anchors = vanilla_anchor_list
            self.yolo_masks = vanilla_anchor_mask
        else:
            anchor_list = net_info["anchors"]
            self.yolo_masks = [[int(y) for y in x.split(',')] for x in net_info["yolo_masks"].split('|')]
            self.anchors  = [[float(y) for y in x.split(',')] for x in anchor_list.split("'")[0].split('|')]
        ########################
        self.num_anchors = len(self.anchors)
        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            a = torch.cuda.FloatTensor().cuda()

    def process(self,trt_output):
        with torch.no_grad():
            # after process
            p1_time = time.time()
            write = 0
            for output, shape, anchor in zip(trt_output, self.output_shapes, self.anchors):
                output = output.reshape(shape)
                trt_output = torch.from_numpy(output).cuda().data
                # print(trt_output.shape)
                # print(anchor)
                trt_output = predict_transform(trt_output, self.inp_dim, anchor, self.num_classes,)
                # print("inp_dim:",self.inp_dim)
                if type(trt_output) == int:
                    continue
                if not write:
                    outputs = trt_output
                    write = 1
                else:
                    outputs = torch.cat((outputs, trt_output), 1)
            p2_time = time.time()
            print("trt_output reshape time:",(p2_time-p1_time))

            # nms 
            for detections in outputs:
                print("detections shape:",detections.shape)
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
            
            return (main_box_corner,main_clss,main_clss_prob) 
        
    def img_size(self):
        return self.inp_w,self.inp_h   


class TrtYOLO():
    def __init__(self,engine_file):
        self.engine_file = engine_file
       
        
        self.engine = get_engine(self.engine_file)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        print("Create tensorRT engine success")

    def do_inference(self,img):
        inference_start = time.time()
        print(self.context)
        self.inputs[0].host = img
        #trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
         # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
         # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        
        inference_end = time.time() 
        return  [out.host for out in self.outputs]



class trtYOLO():
    def __init__(self,cfg_file,engine_file,vanilla_anchor=True):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            a = torch.cuda.FloatTensor().cuda()
        # model param 
        self.cfg_file = cfg_file
        self.moduel_def = parse_cfg(cfg_file)
        self.net_info = self.moduel_def[0]
        self.inp_w = int(self.net_info['width'])
        self.inp_h = int(self.net_info['height'])
        self.inp_dim = self.inp_h
        self.nms_thres = float(self.net_info['nms_thres'])
        self.conf_thres = float(self.net_info['conf_thres'])
        self.num_classes =int( self.net_info['num_classes'])
        self.output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)] #yolo3-608
        self.yolo_anchors = [[(116, 90), (156, 198), (373, 326)],
                             [(30, 61),  (62, 45),   (59, 119)],
                             [(10, 13),  (16, 30),   (33, 23)]]
        ####### anchors TODO
        if vanilla_anchor:
            self.anchors = vanilla_anchor_list
            self.yolo_masks = vanilla_anchor_mask
        else:
            anchor_list = net_info["anchors"]
            self.yolo_masks = [[int(y) for y in x.split(',')] for x in net_info["yolo_masks"].split('|')]
            self.anchors  = [[float(y) for y in x.split(',')] for x in anchor_list.split("'")[0].split('|')]
        ########################
        self.num_anchors = len(self.anchors)
        
        
        
        # tensorrt engine 
        self.engine_file = engine_file
        self.engine = get_engine(self.engine_file)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        print("Create tensorRT engine success")

    def img_size(self):
        return self.inp_w,self.inp_h   

    def detection(self,img_array):
        with torch.no_grad():
            inference_start = time.time()
            print(self.context)
            self.inputs[0].host = img_array
            #trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
             # Run inference.
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
             # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            # Synchronize the stream
            self.stream.synchronize()
            trt_outputs = [out.host for out in self.outputs]
            inference_end = time.time() 
            print("inferencce time:",(inference_end-inference_start))


            write = 0
            for output, shape, anchors in zip(trt_outputs, self.output_shapes, self.yolo_anchors):
                output = output.reshape(shape)
                trt_output = torch.from_numpy(output).cuda().data
                # print(trt_output.shape)
                trt_output = predict_transform(trt_output, self.inp_dim, anchors, self.num_classes, self.use_cuda)
                print("inp_dim:",self.inp_dim)
                if type(trt_output) == int:
                    continue
                if not write:
                    detections = trt_output
                    write = 1
                else:
                    detections = torch.cat((detections, trt_output), 1)

            reshape_time = time.time()
            print("reshape time:",(reshape_time - inference_end))

            # nms 
            for detection in detections:
                print("detection shape:",detection.shape)
                detection = detection[detection[:,4]>self.conf_thres]
                box_conrner = torch.zeros((detection.shape[0],4),device=detection.device)
                xy = detection[:,0:2]
                wh = detection[:,2:4]/2
                box_conrner[:,0:2] = xy-wh
                box_conrner[:,2:4] = xy+wh
                probabilities = detection[:,4]
                clss_prob,clss = torch.max(detection[:,5:],1).data.numpy()
                nms_indices = nms(box_conrner,probabilities,self.nms_thres)
                main_box_corner = box_corner[nms_indices]
                main_clss = clss[nms_indices]
                main_clss_prob = clss_prob[nms_indices]
                if nms_indices.shape[0] == 0:
                    continue
            
            return (main_box_corner,main_clss,main_clss_prob) 