#!/home/blackwidow/catkin_ws/src/multi_lane_detection/venv38/bin/python

import rospy
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX

class image_converter:
    
    def __init__(self) -> None:
        rospy.init_node('video_frame', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/rs_camera/aligned_depth_to_color/image_raw", Image ,self.callback)
        self.image = np.zeros((480, 640), dtype='uint8')
        self.dst_folder = rospy.get_param('~dst_folder','test_folder') # the name of the base frame of the robot
        self.rate = rospy.Rate(15)
        
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            cv_image = cv2.normalize(cv_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            self.image = cv_image
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
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