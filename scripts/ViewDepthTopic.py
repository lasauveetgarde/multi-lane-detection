#!/home/katya/catkin_ws/src/multi-lane-detection/venv38/bin/python

import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import pyrealsense2 as rs2


font = cv2.FONT_HERSHEY_SIMPLEX

class image_converter:
    
    def __init__(self) -> None:
        rospy.init_node('video_frame', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(depth_image_topic, Image ,self.callback)
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = rospy.Subscriber(confidence_topic, Image, self.confidenceCallback)

        self.intrinsics = None
        self.pix = None
        self.pix_grade = None

        self.image = np.zeros((480, 640), dtype='uint8')
        # self.dst_folder = rospy.get_param('~dst_folder','test_folder') # the name of the base frame of the robot
        self.rate = rospy.Rate(15)
        
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_image = cv2.normalize(cv_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            self.image = cv_image

            indices = np.array([[300], [400]])[:,0]
            pix = (indices[1], indices[0])
            self.pix = pix
            print(cv_image[pix[1], pix[0]])


        except CvBridgeError as e:
            print(e)

    def confidenceCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            grades = np.bitwise_and(cv_image >> 4, 0x0f)
            if (self.pix):
                self.pix_grade = grades[self.pix[1], self.pix[0]]
        except CvBridgeError as e:
            print(e)
            return  

if __name__ == '__main__':
    depth_image_topic = '/rs_camera/aligned_depth_to_color/image_raw'

    try:        
        reciev_image = image_converter()
        rate = reciev_image.rate        
        while not rospy.is_shutdown():              
            frame = reciev_image.image 
            rate.sleep()
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
      rospy.loginfo(f'EXCEPTION CATCHED:\n {e}')
    cv2.destroyAllWindows()