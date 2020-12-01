import sys
import argparse
import yaml
import os
import cv2
import torch

from torch.autograd import Variable
from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def preprocess_image(image_path, imgsize, gpu=-1):
    '''
    Read and pre-process the image.
    '''
    # Load image
    img = cv2.imread(image_path)

    # Preprocess image
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    img = np.transpose(img / 255., (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)

    if gpu >= 0:
        # Send model to GPU
        img = Variable(img.type(torch.cuda.FloatTensor))
    else:
        img = Variable(img.type(torch.FloatTensor))

    return img


def main():
    '''
    Main function of test an image.
    '''
    # Choose config file for this demo
    cfg_path = './config/gaussian_yolov3_eval.cfg'

    # Specify checkpoint file which contains the weight of the model you want to use
    ckpt_path = './weights/gaussian_yolov3_coco.pth'

    # Path to the image file fo the demo
    image_path = './data/gaussian_yolov3/traffic_1.jpg'

    # Detection threshold
    detect_thresh = 0.3

    # Use CPU if gpu < 0 else use GPU
    gpu = 1

    # Load configs
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f)

    model_config = cfg['MODEL']
    imgsize = cfg['TEST']['IMGSIZE']
    nmsthre = cfg['TEST']['NMSTHRE']
    gaussian = cfg['MODEL']['GAUSSIAN']

    # if detect_thresh is not specified, the parameter defined in config file is used
    if detect_thresh:
        confthre = detect_thresh
    else:
        confthre = cfg['TEST']['CONFTHRE']

    # Load model
    model = YOLOv3(model_config)

    # Load weight
    state = torch.load(ckpt_path)
    if 'model_state_dict' in state.keys():
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    # a switch for some specific layers/parts that behave differently during training and inference
    model.eval()

    # send your model to the "current device"
    if gpu >= 0:
        model.cuda()

    # load image
    img = preprocess_image(image_path, imgsize, gpu)

    # Inference and postprocess
    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, 80, confthre, nmsthre)

    if outputs[0] is None:
        print("No Objects Deteted!!")
        sys.exit(0)


if __name__ == '__main__':
    main()
