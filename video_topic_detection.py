#!/home/blackwidow/catkin_ws/src/multi_lane_detection/venv38/bin/python3 


import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import os
import time

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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

class image_converter:
    
    def __init__(self) -> None:
        rospy.init_node('video_frame', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/cam0/image_raw",Image,self.callback)
        self.image = np.zeros((480, 640, 3), dtype='uint8')
        self.dst_folder = rospy.get_param('~dst_folder','test_folder') # the name of the base frame of the robot
        self.rate = rospy.Rate(15)
        
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.image = cv_image

if __name__ == '__main__':
    try:        
        reciev_image = image_converter()
        rate = reciev_image.rate        
        while not rospy.is_shutdown():              
            frame = reciev_image.image

            image = frame
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(image)
            orig_image = image.copy()
            image = transform(image)
            image = image.unsqueeze(0).to(device)
            masks, boxes, labels = get_outputs(image, model, threshold)

            # Create a black image for drawing the segmentation map
            black_image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Call the modified draw_segmentation_map function with both orig_image and black_image
            result = draw_segmentation_map(orig_image, masks, boxes, labels, args, background=black_image)
            rate.sleep()

            cv2.imshow('Original Image', frame)
            cv2.imshow('Segmentation Map', np.array(result))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
      rospy.loginfo(f'EXCEPTION CATCHED:\n {e}')
    cv2.destroyAllWindows()
