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
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)

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

            indices = np.array([[161], [72]])[:,0]
            pix = (indices[1], indices[0])
            self.pix = pix
            # print(cv_image[pix[1], pix[0]])

            if self.intrinsics:
                depth = cv_image[pix[1], pix[0]]
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                x_coordinate = result[0]/1000
                y_coordinate = result[1]/1000
                z_coordinate = result[2]/1000
                print('Coordinate: %8.2f %8.2f %8.2f.' % (result[0]/1000, result[1]/1000, result[2]/1000)
)
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

if __name__ == '__main__':
    depth_image_topic = '/rs_camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/rs_camera/aligned_depth_to_color/camera_info'

    try:        
        reciev_image = image_converter()
        rate = reciev_image.rate        
        while not rospy.is_shutdown():              
            frame = reciev_image.image 
            rate.sleep()
            image = cv2.circle(frame, (300,400), radius=10, color=(0, 0, 255), thickness=-1)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
      rospy.loginfo(f'EXCEPTION CATCHED:\n {e}')
    cv2.destroyAllWindows()