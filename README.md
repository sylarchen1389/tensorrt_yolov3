# tensorrt_yolov3
------
## Description
Use nvidia tensorRT to speed up yolov3 inference

## Requirements:
- CUDA>=10.0
- pytorch>=1.0
- pycuda
- opencv_python
- TensorRT  5.0.2.6/7


## Installation Instructions:

### CUDA:
CUDA is an NVIDIA GPU programming language, with installation instructions
that can be found on NVIDIA's website at:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### TensorRT:
TensorRT is a CUDA-based deep learning inference platform optimized for NVIDIA
GPU's, used to run cone detections and keypoint detections on the MIT/DUT18D
car. TensorRT installation instructions can also be found on NVIDIA's website
at:
https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html

### Yolo v3 
official website: [YOLO v3](https://github.com/pjreddie/darknet) 


## Usage
## 1. Download darknet weights and model cfg

weightsfile only for COCO [here](https://pjreddie.com/media/files/yolov3.weights),

Or, you could just type (if you're on Linux)
```
wget https://pjreddie.com/media/files/yolov3.weights  
```

## 2. Convert the darknet weights to onnx
!!! only support python2 
```
python darknet to onnx.py
```

## 3. Covert onnx to TensorRT engine file:
chage ONNX_FILE_PATH and ENGINE_FILE_PATH in onnx_to_tensorrt.py and run:
```
python3 onnx_to_tensorrt.py
```
## 4. Detect
```
python detect --engine_path <your engine file> --target_path <image>
```
optional arguments:
```
  -h, --help            show this help message and exit
  --cfg_path CFG_PATH
  --target_path TARGET_PATH
                        path to target image/video
  --engine_file ENGINE_FILE
                        path to tensortRT engine file
  --mode MODE           detect: image or video
  --camera_device CAMERA_DEVICE
                        code of camera device
  --vanilla_anchor      whether to use vanilla anchor boxes 
  --no_vanilla_anchor   
```
example:
```
python detect.py --engine_path yolov3-608.trt --target_path data/dog.jpg
```

## 5. On a Camera
```
python dectect.py --camera_device <code>  --mode video
```
example:
```
python dectect.py --camera_device 0  --mode video
```

## 6. for int8 model 
this model inference with fp32

the int8 one is writing