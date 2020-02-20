
from models import PrepocessYOLO,TrtYOLO








def main():
    prepocessor = PrepocessYOLO()
    trt_model = TrtYOLO()
    detect()

def detect():

def detect_video():

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--model_cfg', type=str, default='model_cfg/yolo_baseline.cfg')
    parser.add_argument('--target_path', type=str,default='images/dog.jpg', help='path to target image/video')
    parser.add_argument('--output_path', type=str, default="outputs/visualization/")
    parser.add_argument('--weights_path', type=str, default='yolov3.weights',help='path to weights file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.25, help='IoU threshold for non-maximum suppression')
    parser.add_argument('--engine_path',type=str,default='yolov3.trt',help='path to tensortRT engine file')

    add_bool_arg('vanilla_anchor', default=False, help="whether to use vanilla anchor boxes for training")
    ##### Loss Constants #####
    parser.add_argument('--xy_loss', type=float, default=2, help='confidence loss for x and y')
    parser.add_argument('--wh_loss', type=float, default=1.6, help='confidence loss for width and height')
    parser.add_argument('--no_object_loss', type=float, default=25, help='confidence loss for background')
    parser.add_argument('--object_loss', type=float, default=0.1, help='confidence loss for foreground')

    opt = parser.parse_args()

    