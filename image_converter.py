#!/home/katya/catkin_ws/src/multi-lane-detection/venv38/bin/python

import numpy as np
import torch.nn as nn
import os
import time

from PIL import Image
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
    
    def __init__(self) -> None:
        # rospy.init_node('video_frame', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/rs_camera/color/image_raw",Image,self.RGBImageCallback)
        self.image_sub = rospy.Subscriber("/rs_camera/aligned_depth_to_color/image_raw", Image ,self.DepthImageCallback)

        self.image = np.zeros((480, 640, 3), dtype='uint8')
        self.depth_image = np.zeros((480, 640, 3), dtype='uint8')

        self.dst_folder = rospy.get_param('~dst_folder','test_folder') # the name of the base frame of the robot
        self.rate = rospy.Rate(10)
        
    def RGBImageCallback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.image = cv_image

    def DepthImageCallback(self,data):
        try:
            depth_cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_image = cv2.normalize(cv_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            # print(cv_image[pix[1], pix[0]])
        except CvBridgeError as e:
            print(e)
        self.depth_image = depth_cv_image