#!/home/katya/catkin_ws/src/multi-lane-detection/venv38/bin/python

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs2

depth_image_topic = '/rs_camera/aligned_depth_to_color/image_raw'
depth_info_topic = '/rs_camera/aligned_depth_to_color/camera_info'

class CountPoints():
    
    def __init__(self) -> None:
        # rospy.init_node('video_frame', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(depth_image_topic, Image ,self.callback)
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = rospy.Subscriber(confidence_topic, Image, self.confidenceCallback)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)

        self.intrinsics = None
        self.pix = [0, 0]
        self.pix_grade = None

        self.image = np.zeros((480, 640), dtype='uint8')
        self.rate = rospy.Rate(15)

        self.three_d_coordinate = []
        self.result_array = [0,0,0]

    def set_res_arr(self, result_array):
        self.result_array = result_array
        

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            self.image = cv_image

            self.three_d_coordinate = []
            for i in range(len(self.result_array)):
                indices = np.array([[self.result_array[i][0]],[self.result_array[i][1]]])[:,0]
                pix = (indices[1], indices[0])
                if self.intrinsics:
                    depth = cv_image[int(pix[0])-1, int(pix[1])-1]
                    result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                    self.three_d_coordinate.append([result[0]/1000, result[1]/1000, result[2]/1000])
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

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return