#!/home/katya/catkin_ws/src/multi-lane-detection/venv38/bin/python


import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import os
import time

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs, array_segmentation_map
from image_converter import image_converter
from count_3dpoint import CountPoints
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

import rospy
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

font = cv2.FONT_HERSHEY_SIMPLEX

parser = argparse.ArgumentParser()
parser.add_argument(
    '--no-boxes',
    action='store_true',
    help='do not show bounding boxes, only show segmentation map'
)
args = parser.parse_args()

# VIDEO_PATH = 'input/inference_data/video_3.mp4'
# VIDEO_PATH = 0

WEIGHTS_PATH = 'outputs/training/road_line/model_15.pth'
threshold = 0.9

OUT_DIR = os.path.join('outputs', 'inference')
os.makedirs(OUT_DIR, exist_ok=True)

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    pretrained=False, num_classes=91
)

model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

ckpt = torch.load(WEIGHTS_PATH)
model.load_state_dict(ckpt['model'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def publishPC2(count,xyz):
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
    ]

    header = Header()
    header.frame_id = "base_link"
    header.stamp = rospy.Time.now()

    # x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2, 2, height))
    # z = 0.5 * np.sin(2 * x - count / 10.0) * np.sin(2 * y)
    x=xyz[0]
    y=xyz[1]
    # z=xyz[2]
    points = np.array([x, y, 0, 0]).reshape(4, -1).T

    pc2 = point_cloud2.create_cloud(header, fields, points)
    pub.publish(pc2)


if __name__ == '__main__':

    rospy.init_node("pc2_publisher")
    pub = rospy.Publisher("points2", PointCloud2, queue_size=100)
    rate = rospy.Rate(10)
    count = 0

    # try:        
    reciev_image = image_converter()
    rate = reciev_image.rate 
    thrd_point = CountPoints()

    while not rospy.is_shutdown():              
        frame = reciev_image.image
        depth_frame = reciev_image.depth_image
        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_image = image.copy()
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        masks, boxes, labels = get_outputs(image, model, threshold)

        # background= black_image IF WANT TO VIEW COLORED DETECTION
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Call the modified draw_segmentation_map function with both orig_image and black_image
        result = draw_segmentation_map(depth_frame, masks, boxes, labels, args, background=None)
        result_array = array_segmentation_map(depth_frame, masks, boxes, labels)
        # print(f'array is {result_array}')
        # print(f'condition is {result_array!=[]} and {result_array is not None}')
        if result_array!=[] and result_array is not None:
            thrd_point.set_res_arr(result_array=result_array)
            xyz=thrd_point.three_d_coordinate
            print(xyz)
            publishPC2(count,xyz)
            count += 1
            # rate.sleep()

        rate.sleep()

        cv2.imshow('Original Image', frame)
        cv2.imshow('Segmentation Map', np.array(result))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # except Exception as e:
    #   rospy.loginfo(f'EXCEPTION CATCHED:\n {e}')
    cv2.destroyAllWindows()
