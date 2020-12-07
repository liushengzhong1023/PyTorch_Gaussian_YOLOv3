import sys
import argparse
import yaml
import json
import os
import cv2
import time
import torch

from torch.autograd import Variable
from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox
from utils.IoU import get_iou

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

waymo_classes = {0: 'unknown', 1: 'vehicle', 2: 'pedestrian', 3: 'sign', 4: 'cyclist'}
waymo_to_coco = {0: 10, 1: 2, 2: 0, 3: 11, 4: 1}  # from waymo to coco

# -------------------------------------------- COCO Util Functions ----------------------------------
coco_class_names = (  # 'background',  # class zero
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'car',  # 'truck',
    'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
    'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

coco_class_colors = np.empty(shape=(0, 3), dtype=np.int)
palette = sns.color_palette(n_colors=len(coco_class_names))
for color in palette:
    r, g, b = color[0] * 255, color[1] * 255, color[2] * 255
    rgb = np.array([int(r), int(g), int(b)])
    coco_class_colors = np.append(coco_class_colors, rgb[None, :], axis=0)


# -------------------------------------------- Waymo Util Functions ----------------------------------
def read_segment_labels(segment_path):
    '''
    Load the segment labels, including bbox attributes, class information, and object id.
    '''
    segment_name = os.path.basename(segment_path)
    json_file = os.path.join(segment_path, segment_name + ".json")

    with open(json_file, 'r') as f:
        segment_labels = json.load(f)

    return segment_labels


def extract_image_time_from_file_name(image_path):
    '''
    Extract the image time from file name.
    '''
    image_base_name = os.path.basename(image_path)
    image_time = image_base_name.split('-')[0]

    return image_time


# -----------------------------------------------------------------------------------------------------


def load_image(image_path, new_size, gpu=-1):
    '''
    Read and pre-process the image. The returned image is within [0, 1] range
    '''
    # Load image
    img = cv2.imread(image_path)

    # Preprocess image, save the raw image in RGB, dimensions [color, h, w]
    # img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img_raw = img.copy()
    img, info_img = preprocess_pad(img, new_size)  # info = (h, w, nh, nw, dx, dy)

    # normalize the image
    img = np.transpose(img / 255., (2, 0, 1))

    # convert to PyTorch tensor and add a new batch dimension
    img = torch.from_numpy(img).float().unsqueeze(0)

    # decide the tensor type according to the device used
    if gpu >= 0:
        img = Variable(img.type(torch.cuda.FloatTensor))
    else:
        img = Variable(img.type(torch.FloatTensor))

    return img, img_raw, info_img


def main():
    '''
    Main function of test an image.
    '''
    # -------------------------------------------- Load Model Config ----------------------------------
    # Choose config file for this demo
    cfg_path = './config/gaussian_yolov3_eval.cfg'

    # Specify checkpoint file which contains the weight of the model you want to use
    ckpt_path = './weights/gaussian_yolov3_coco.pth'

    # Path to the image file fo the demo
    validation_path = '/home/sl29/data/Waymo/validation_images'
    output_path = '/home/sl29/DeepScheduling/src/temporal_locality/PyTorch_Gaussian_YOLOv3/waymo_detections/full_frame_detections'
    segment_list = os.listdir(validation_path)

    # Detection threshold
    detect_thresh = 0.5

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

    # -------------------------------------------- Load YOLOv3 Model ----------------------------------
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

    # ----------------------------------- Load Waymo Data & Inference ----------------------------------
    for segment in segment_list:
        print("Evaluating: " + segment)
        start = time.time()

        # input and output
        segment_path = os.path.join(validation_path, segment)
        output_file = os.path.join(output_path, segment + '.json')

        # read the labels
        segment_labels = read_segment_labels(segment_path)
        segment_detections = dict()

        # dx, dy is the original image shift in the padded image
        h, w, nh, nw, dx, dy = (1280, 1920, 1280, 1920, 0, 320)

        # image name list
        image_name_list = []
        for e in os.listdir(segment_path):
            if e.endswith('.jpeg'):
                image_name_list.append(e)
        image_name_list.sort()

        for image_name in image_name_list:
            # read image
            image_file = os.path.join(segment_path, image_name)

            # read labels
            image_time = extract_image_time_from_file_name(image_name)
            image_bbox_list = segment_labels[image_time]['FRONT']

            # load image
            img, img_raw, info_img = load_image(image_file, 1920, gpu)

            # Inference and postprocess
            with torch.no_grad():
                outputs = model(img)
                outputs = postprocess(outputs, 80, confthre, nmsthre)

            if outputs[0] is None:
                print("No Objects Deteted at time: " + image_name)

            # visualize the detection
            detections = list()

            if outputs[0] is not None:
                for output in outputs[0]:
                    x1, y1, x2, y2, conf, cls_conf, cls_pred = output[:7]
                    x1 = x1.item()
                    y1 = y1.item()
                    x2 = x2.item()
                    y2 = y2.item()
                    cls_pred = cls_pred.item()
                    conf = conf.item()
                    cls_conf = cls_conf.item()

                    # minus the shift
                    y1 -= dy
                    y2 -= dy
                    x1 -= dx
                    x2 -= dx

                    if gaussian:
                        sigma_x, sigma_y, sigma_w, sigma_h = output[7:]
                        mean_sigma = torch.mean(torch.stack([sigma_x, sigma_y, sigma_w, sigma_h])).cpu().numpy().item()

                    # update box list
                    cls_pred = int(cls_pred)
                    box_color = coco_class_colors[cls_pred]
                    detections.append({
                        'box_value': [x1, y1, x2, y2],  # coordinates in the original image
                        'coco_cls_pred': cls_pred,
                        'coco_cls_name': coco_class_names[cls_pred],
                        'objectness_score': conf,
                        'class_conf': cls_conf,
                        'conf_score': cls_conf * conf,
                        'variance': mean_sigma if gaussian else 0,
                        'color': (int(box_color[0]), int(box_color[1]), int(box_color[2]))
                    })

            # save the frame detection results
            segment_detections[image_time] = detections

        # save detection results for this segment
        with open(output_file, 'w') as f:
            f.write(json.dumps(segment_detections, indent=4))

        end = time.time()
        print("Segment execution time: %f s" % (end - start))
        print("------------------------------------------------------------------------")


if __name__ == '__main__':
    main()
