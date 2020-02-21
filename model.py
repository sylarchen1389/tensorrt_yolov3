import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

from PIL import Image, ImageDraw
import torchvision

from utils.utils import *
from utils.parse_config import parse_cfg
from utils.nms import nms

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
    def __init__(self, cfg_file,engine_file,vanilla_anchor = True):
        self.use_cuda = True
        if self.use_cuda:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            a = torch.cuda.FloatTensor().cuda()  #pytorch必须首先占用部分CUD
        
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
        self.anchors = [[(116, 90), (156, 198), (373, 326)],
                             [(30, 61),  (62, 45),   (59, 119)],
                             [(10, 13),  (16, 30),   (33, 23)]]
        ####### anchors TODO
        if  not vanilla_anchor:
            anchor_list = net_info["anchors"]
            self.yolo_masks = [[int(y) for y in x.split(',')] for x in net_info["yolo_masks"].split('|')]
            self.anchors  = [[float(y) for y in x.split(',')] for x in anchor_list.split("'")[0].split('|')]
        ########################
        self.num_anchors = len(self.anchors)

        # tensorRT engine
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
    
    def img_size(self):
        return self.inp_w,self.inp_h

    def detect_path(self,target_path):
        # opencv prep
        im_time1 = time.time()
        img = cv2.imread(target_path)
        c,w,h = img.shape
        new_width, new_height = trt_model.img_size()
        pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
        constant = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)#top,bottom,left,right
        img_ = cv2.resize(constant,(new_height,new_width), interpolation=cv2.INTER_CUBIC)
        img_ = img_[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        img_ = img_.numpy()
        print(img_.shape)
        im_time2 = time.time()
        print("[time]cv prep image time:",(im_time2-im_time1 ))

        main_box_corner,clss,clss_prob = self.detection(img_)

        boxes = torch.empty(size = main_box_corner.shape)

        boxes[:,0] = main_box_corner[:, 0]/ ratio - pad_w
        boxes[:,1] = main_box_corner[:, 1] / ratio - pad_h
        boxes[:,2] = main_box_corner[:, 2] / ratio - pad_w
        boxes[:,3] = main_box_corner[:, 3] / ratio - pad_h 
        boxes = boxes.cpu().numpy()
        clss_prob = clss_prob.cpu().numpy()
        clss = clss.cpu().numpy

        #img_with_boxes = draw_bboxes(img,boxes,clss_prob,clss,ALL_CATEGORIES)
        # img_with_boxes.save(target_path.split('/')[-1])


    def detect_frame(self,frame):
        # opencv prep
        im_time1 = time.time()
        c,w,h = frame.shape
        new_width, new_height = trt_model.img_size()
        pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
        img_ = cv2.copyMakeBorder(frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)#top,bottom,left,right
        img_ = cv2.resize(img_,(new_height,new_width), interpolation=cv2.INTER_CUBIC)
        img_ = img_[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        img_ = img_.numpy()
        print(img_.shape)
        im_time2 = time.time()
        print("[time]cv prep image time:",(im_time2-im_time1 ))

        main_box_corner,clss,clss_prob = self.detection(img_)
        if main_box_corner ==None:
            print("[waring]No object in veiw")
            return None
    
        boxes = torch.empty(size = main_box_corner.shape)

        boxes[:,0] = main_box_corner[:, 0]/ ratio - pad_w
        boxes[:,1] = main_box_corner[:, 1] / ratio - pad_h
        boxes[:,2] = main_box_corner[:, 2] / ratio - pad_w
        boxes[:,3] = main_box_corner[:, 3] / ratio - pad_h 
        boxes = boxes.cpu().numpy()
        clss_prob = clss_prob.cpu().numpy()
        clss = clss.cpu().numpy

        #img_with_boxes = draw_bboxes(img,boxes,clss_prob,clss,ALL_CATEGORIES)
        # img_with_boxes.save(target_path.split('/')[-1])
   
    # 做推断
    def detection(self,img):
        # (img, orig_img, im_name, im_dim_list) = procession_tuple
        with torch.no_grad():
            inference_start = time.time()
            self.inputs[0].host = img
            # trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
             # Run inference.
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
             # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            # Synchronize the stream
            self.stream.synchronize()
            trt_outputs = [out.host for out in self.outputs]
            inference_end = time.time()
            print('[Time]inference time : %f' % (inference_end-inference_start))
           
            write = 0
            for output, shape, anchors in zip(trt_outputs, self.output_shapes, self.anchors):
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
                try:
                    clss_prob,clss = torch.max(detections[:,5:],1)
                except:
                    return None,None,None
                nms_indices = nms(box_corner,probabilities,self.nms_thres)
                main_box_corner = box_corner[nms_indices]
                main_clss = clss[nms_indices]
                main_clss_prob = clss_prob[nms_indices]
                if nms_indices.shape[0] == 0:
                    continue


            o_time3 = time.time()
            print("[Time]nms time:",(o_time3 - o_time1))

        return  (main_box_corner,main_clss,main_clss_prob) 

    

    def process_frame(self, frame_dic):
        pass

    def process_frame_batch(self, frame_dic_list):
        
        
        img_list = []
        for dic in frame_dic_list:
            img_list.append(dic['img'])
        
        procession_tuple = self.preparing(img_list)
        # (img, orig_img, im_name, im_dim_list) = procession_tuple
        (main_box_corner,main_clss,main_clss_prob)  = self.detection(procession_tuple)
        print(main_box_corner,main_clss,main_clss_prob) 
        
        torch.cuda.empty_cache()
        
        return frame_dic_list



# test
if __name__ == '__main__':
    init_dict = {'trt':"yolov3-608.trt", 'use_cuda':True}
    trt_model = trtYOLO('model_cfg/yolov3-608.cfg',"yolov3-608.trt")

    cap = cv2.VideoCapture(1)  # 打开摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    #img_path = './images/person2.jpg'
    #dic = {'img':cv2.imread(img_path),'data':{},'info':{}}
    #input_dic_list.append(dic)

    f_start = time.time()
    frame_count = 0
    while True:
        print("-----------------------------")
        frame_count += 1 
        t_time1 = time.time()
        return_value, frame = cap.read()  # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像        
        t_time2 = time.time()
        print("[Time]get frame time :",(t_time2-t_time1))

        # trt_model.detect('data/dog.jpg')
        trt_model.detect_frame(frame)
        t_time3 = time.time()
        print("[Time]detect time:",(t_time3-t_time2))
        if cv2.waitKey(1) & 0xFF == ord('z'):       # 按q退出
            break
       
        print("total time: ",(t_time3-t_time1))
        cv2.imshow("frame",frame)
        print("FPS: ",1./(t_time3- t_time1))
    
    cap.release() 
    cv2.destroyAllWindows() 
    

