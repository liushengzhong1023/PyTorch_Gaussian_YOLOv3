import sys
import argparse
import yaml
import os
import cv2
import time
import torch

from torch.autograd import Variable
from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_image(image_path, imgsize, gpu=-1):
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

    return img, img_raw, info_img


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

    # load coco classes
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    # Detection threshold
    detect_thresh = 0.3

    # Use CPU if gpu < 0 else use GPU
    gpu = 1

    # Flag for visualize
    visualize_flag = False

    # Load configs
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f)

    model_config = cfg['MODEL']
    # imgsize = cfg['TEST']['IMGSIZE']
    imgsize = 1024
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
    img, img_raw, info_img = load_image(image_path, imgsize, gpu)

    # Inference and postprocess
    with torch.no_grad():
        outputs = model(img)

        # evaluate inference time
        # start = time.time()
        #
        # for _ in range(200):
        #     outputs = model(img)
        #
        # end = time.time()
        # print("Average inference time: " + str((end - start) / 200))
        outputs = postprocess(outputs, 80, confthre, nmsthre)

    if outputs[0] is None:
        print("No Objects Deteted!!")
        sys.exit(0)

    # visualize the detection
    bboxes = list()
    classes = list()
    scores = list()
    colors = list()
    sigmas = list()

    for output in outputs[0]:
        x1, y1, x2, y2, conf, cls_conf, cls_pred = output[:7]
        if gaussian:
            sigma_x, sigma_y, sigma_w, sigma_h = output[7:]
            sigmas.append([sigma_x, sigma_y, sigma_w, sigma_h])
            print(torch.mean(torch.stack([sigma_x, sigma_y, sigma_w, sigma_h])))

        cls_id = coco_class_ids[int(cls_pred)]
        box = yolobox2label([y1, x1, y2, x2], info_img)

        # update box list
        bboxes.append(box)
        classes.append(cls_id)
        scores.append(cls_conf * conf)
        colors.append(coco_class_colors[int(cls_pred)])

        # image size scale used for sigma visualization
        h, w, nh, nw, _, _ = info_img
        sigma_scale_img = (w / nw, h / nh)

    if visualize_flag:
        fig, ax = vis_bbox(
            img_raw, bboxes, label=classes, score=scores, label_names=coco_class_names, sigma=sigmas,
            sigma_scale_img=sigma_scale_img,
            sigma_scale_xy=2., sigma_scale_wh=2.,  # 2-sigma
            show_inner_bound=False,  # do not show inner rectangle for simplicity
            instance_colors=colors, linewidth=3)


if __name__ == '__main__':
    main()
