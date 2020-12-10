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


def test_temporal_correlation_with_groundtruth(image_time_list, segmentation_detections, segmentation_labels):
    '''
    Test the correlation across frames in a vide segment.
    '''
    object_list = dict()

    for image_time in image_time_list:
        gt_boxes = segmentation_labels[image_time]['FRONT']
        detected_boxes = segmentation_detections[image_time]

        for detected_box in detected_boxes:
            dt_class = detected_box['coco_cls_pred']
            dt_class_name = coco_class_names[dt_class]
            dt_x1, dt_y1, dt_x2, dt_y2 = detected_box['box_value']
            dt_conf_score = detected_box['conf_score']
            dt_variance = detected_box['variance']
            matched_flag = False

            for gt_box in gt_boxes:
                gt_class = waymo_to_coco[int(gt_box['class'])]
                gt_x1, gt_y1, gt_x2, gt_y2 = gt_box['value']
                gt_object_id = gt_box['object_id']

                # compute IoU score
                iou = get_iou({'x1': dt_x1, 'y1': dt_y1, 'x2': dt_x2, 'y2': dt_y2},
                              {'x1': gt_x1, 'y1': gt_y1, 'x2': gt_x2, 'y2': gt_y2})

                if iou > 0.7:
                    if matched_flag:
                        print("The detected box is matched with two gt boxes!")

                    if gt_object_id not in object_list:
                        object_list[gt_object_id] = dict()
                        object_list[gt_object_id]['gt_class'] = coco_class_names[gt_class]
                        object_list[gt_object_id]['gt_appearances'] = 0
                        object_list[gt_object_id]['conf_list'] = []
                        object_list[gt_object_id]['variance_list'] = []
                        object_list[gt_object_id]['predicted_class_list'] = []

                    object_list[gt_object_id]['conf_list'].append(dt_conf_score)
                    object_list[gt_object_id]['variance_list'].append(dt_variance)
                    object_list[gt_object_id]['predicted_class_list'].append(dt_class_name)
                    matched_flag = True

        # count groundtruth box apeearance
        for gt_box in gt_boxes:
            gt_object_id = gt_box['object_id']
            gt_class = waymo_to_coco[int(gt_box['class'])]

            if gt_object_id not in object_list:
                object_list[gt_object_id] = dict()
                object_list[gt_object_id]['gt_class'] = coco_class_names[gt_class]
                object_list[gt_object_id]['gt_appearances'] = 0
                object_list[gt_object_id]['conf_list'] = []
                object_list[gt_object_id]['variance_list'] = []
                object_list[gt_object_id]['predicted_class_list'] = []

            object_list[gt_object_id]['gt_appearances'] += 1

    # analyze the object prediction confs
    conf_var_list = []
    variance_var_list = []

    for object_id in object_list:
        print(object_id)
        print("Groundtruth class: " + object_list[object_id]['gt_class'])
        print("Groundtruth appearance times: %s" % object_list[object_id]['gt_appearances'])
        print("Detected times: %s" % len(object_list[object_id]['predicted_class_list']))
        print("Prediction class list: ")
        print(object_list[object_id]['predicted_class_list'])

        if not object_list[object_id]['conf_list']:
            pass
        else:
            conf_var = np.var(object_list[object_id]['conf_list'])
            variance_var = np.var(object_list[object_id]['variance_list'])
            conf_var_list.append(conf_var)
            variance_var_list.append(variance_var)

            print("Mean prediction confidence: %f" % np.mean(object_list[object_id]['conf_list']))
            print("Prediction confidence var: %f" % conf_var)
            print("Mean regression variance: %f" % np.mean(object_list[object_id]['variance_list']))
            print("Location regression var: %f" % variance_var)
        print()

    print("------------------------------------------------------------------------")
    mean_conf_var = np.mean(conf_var_list)
    mean_variance_var = np.mean(variance_var_list)
    print("Mean of prediction confidence var on object: %f" % mean_conf_var)
    print("Mean of location regression var: %f" % mean_variance_var)


def main():
    '''
    Main function of test an image.
    '''
    start = time.time()
    # -------------------------------------------- Load Model Config ----------------------------------
    # Path to the image file fo the demo
    validation_path = '/home/sl29/data/Waymo/validation_images'
    # segment_name = 'segment-16751706457322889693'
    segment_name = 'segment-4816728784073043251'
    segment_path = os.path.join(validation_path, segment_name)

    # detection path
    detection_path = '/home/sl29/DeepScheduling/src/temporal_locality/PyTorch_Gaussian_YOLOv3/' \
                     'waymo_detections/full_frame_detections'
    segment_detection_file = os.path.join(detection_path, segment_name + '.json')
    with open(segment_detection_file, 'r') as f:
        segment_detections = json.load(f)

    # ----------------------------------- Load Waymo Data & Inference ----------------------------------
    # read the labels
    segment_labels = read_segment_labels(segment_path)

    # image name list
    image_time_list = list(segment_labels.keys())
    image_time_list.sort()

    # test the temporal correlation
    test_temporal_correlation_with_groundtruth(image_time_list, segment_detections, segment_labels)

    end = time.time()
    print("------------------------------------------------------------------------")
    print("Segment execution time: %f s" % (end - start))


if __name__ == '__main__':
    main()
