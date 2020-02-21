import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
# from base_module import BaseModule
from utils.utils import *
from utils.parse_config import parse_cfg
from utils.nms import nms
#from alpha_yolo3_module_drawing import drawing

# from data_processing import PreprocessYOLO

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


def prep_image(orig_im, inp_dim):
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy() #(3 608 608)
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    img_ = img_.numpy()
    return img_, orig_im, dim

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas

class trtYOLO():
    def __init__(self, cfg_file,engine_file):
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        a = torch.cuda.FloatTensor().cuda()  #pytorch必须首先占用部分CUDA

        self.use_cuda = True
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
        # if vanilla_anchor:
        #     self.anchors = vanilla_anchor_list
        #     self.yolo_masks = vanilla_anchor_mask
        # else:
        #     anchor_list = net_info["anchors"]
        #     self.yolo_masks = [[int(y) for y in x.split(',')] for x in net_info["yolo_masks"].split('|')]
        #     self.anchors  = [[float(y) for y in x.split(',')] for x in anchor_list.split("'")[0].split('|')]
        ########################
        #self.num_anchors = len(self.anchors)

        self.engine = get_engine(engine_file)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

       


    # 处理图片
    def preparing(self,orig_img_list):
        img = []
        orig_img = []
        im_name = []
        im_dim_list = []
        batch = 1
        for im in orig_img_list:
            im_name_k = ''
            img_k, orig_img_k, im_dim_list_k = prep_image(im, self.inp_dim)
            img.append(img_k)
            orig_img.append(orig_img_k)
            im_name.append(im_name_k)
            im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
            im_dim_list_ = im_dim_list

        procession_tuple = (img, orig_img, im_name, im_dim_list)
        return procession_tuple

    # 做推断
    def detection(self,procession_tuple):
        (img, orig_img, im_name, im_dim_list) = procession_tuple
        # with get_engine(self.trt_file) as engine, engine.create_execution_context() as context:
        with torch.no_grad():
            #########################################################################做推断
            # inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
            inference_start = time.time()
            # host？？？ 
            self.inputs[0].host = img[0] #waiting fix bug ？？？？wtf
            
            trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
            inference_end = time.time()
            print('[Time]inference time : %f' % (inference_end-inference_start))
           ##########################################################################
           
            write = 0
            for output, shape, anchors in zip(trt_outputs, self.output_shapes, self.yolo_anchors):
                output = output.reshape(shape)
                trt_output = torch.from_numpy(output).cuda().data
                # print(trt_output.shape)
                trt_output = predict_transform(trt_output, self.inp_dim, anchors, self.num_classes, self.use_cuda)
                # print("inp_dim:",self.inp_dim)
                if type(trt_output) == int:
                    continue
                if not write:
                    outputs = trt_output
                    write = 1
                else:
                    outputs = torch.cat((outputs, trt_output), 1)

            #print("trt_output shape: " ,trt_o.shape )
           
            # print("detections shape:",detections.shape)
            o_time1 = time.time()
            print("[Time]Reshape time:",(o_time1 - inference_end))
            print('[Time]TensorRT inference time : %f' % (o_time1-inference_start))
            
            for detections in outputs:
                # print("detections shape:",detections.shape)
                detections = detections[detections[:,4]>self.conf_thres]
                box_corner = torch.zeros((detections.shape[0],4),device=detections.device)
                xy = detections[:,0:2]
                wh = detections[:,2:4]/2
                box_corner[:,0:2] = xy-wh
                box_corner[:,2:4] = xy+wh
                probabilities = detections[:,4]
                clss_prob,clss = torch.max(detections[:,5:],1)
                nms_indices = nms(box_corner,probabilities,self.nms_thres)
                main_box_corner = box_corner[nms_indices]
                main_clss = clss[nms_indices]
                main_clss_prob = clss_prob[nms_indices]
                if nms_indices.shape[0] == 0:
                    continue


            o_time3 = time.time()
            print("[Time]nms time:",(o_time3 - o_time1))

        return  (main_box_corner,main_clss,main_clss_prob) 

    def dict_checkup(self,dict):
        if 'img' not in dict:
            dict['img']= ''
            print('no img in dict')	
        if 'data' not in dict:
            dict['data']={}
            print('no data in dict')
        if 'info' not in dict:
            dict['info']={}
            print('no info in dict')	

    def process_frame(self, frame_dic):
        pass

    def process_frame_batch(self, frame_dic_list):
        for dic in frame_dic_list:
            self.dict_checkup(dic)
        
        img_list = []
        for dic in frame_dic_list:
            img_list.append(dic['img'])
        
        procession_tuple = self.preparing(img_list)
        # (img, orig_img, im_name, im_dim_list) = procession_tuple
        (main_box_corner,main_clss,main_clss_prob)  = self.detection(procession_tuple)
        print(main_box_corner,main_clss,main_clss_prob) 
        if len(class_list_all) == 0:
            for frame_dic in frame_dic_list:
                frame_dic['data']['number'] = 0
                frame_dic['data']['box_list'] = []
                frame_dic['data']['class_list'] = []
                frame_dic['data']['conf_list'] = []
        else:
            for i,frame_dic in enumerate(frame_dic_list):
                frame_dic['data']['number'] = len(class_list_all[i])
                frame_dic['data']['box_list'] = box_list_all[i]
                frame_dic['data']['class_list'] = class_list_all[i]
                frame_dic['data']['conf_list'] = conf_list_all[i]

        torch.cuda.empty_cache()
        
        return frame_dic_list





if __name__ == '__main__':
    init_dict = {'trt':"yolov3-608.trt", 'use_cuda':True}
    alpha_yolo3_unit = trtYOLO('model_cfg/yolov3-608.cfg',"yolov3-608.trt")

    cap = cv2.VideoCapture(0)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)()
    #img_path = './images/person2.jpg'
    #dic = {'img':cv2.imread(img_path),'data':{},'info':{}}
    #input_dic_list.append(dic)

    f_start = time.time()
    frame_count = 0
    while True:
        frame_count += 1 
        t_start = time.time()
        return_value, frame = cap.read()  # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像        
        t_end = time.time()
        print("get frame time :",(t_end-t_start))

        #print(" node get image ",str(datetime.datetime.now()))
        if return_value is False:
            print('read video error')
            exit()

        input_dic_list = []
        input_dic = {'img':frame,'data':{},'info':{}}
        input_dic_list.append(input_dic)

        detect_start = time.time()
        output_dic_list = alpha_yolo3_unit.process_frame_batch(input_dic_list)
        detect_end = time.time()
        
        print("detect time :",(detect_end - detect_start))
        
        for dic in output_dic_list:
            img_array = dic['img']
            drawing(img_array,dic)	
            frame = np.array(img_array)
            print("frame shape: ",frame.shape)
            cv2.imshow('frame',img_array)
            #cv2.waitKey(5000)
        if cv2.waitKey(1) & 0xFF == ord('z'):       # 按q退出
            break
       
        t_end = time.time()
        print("total time: ",(t_end-t_start))
        print("FPS: ",frame_count/(t_end- f_start))
    
    cap.release() 
    cv2.destroyAllWindows() 
    

