#!/home/blackwidow/catkin_ws/src/multi_lane_detection/venv38/bin/python3 

import rospy
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

width = 100
height = 100


def publishPC2(count):
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
    ]

    header = Header()
    header.frame_id = "map"
    header.stamp = rospy.Time.now()

    x = []
    y = []
    test_array = [ (145.0, 107), (145.0, 112), (145.0, 117), (145.0, 122), (145.0, 127), (145.0, 132),...
                   (145.0, 137), (145.0, 142), (145.0, 147), (145.0, 152), (145.0, 157), (145.0, 162),...
                   (145.0, 167), (145.0, 172), (145.0, 177), (145.0, 182), (145.0, 187), (145.0, 192),...
                   (145.0, 197), (145.0, 202), (145.0, 207), (145.0, 212), (145.0, 217), (145.0, 222),...
                   (145.0, 227), (145.0, 232), (145.0, 237), (145.0, 242), (145.0, 247), (145.0, 252),...
                   (145.0, 257), (145.0, 262), (145.0, 267), (145.0, 272), (145.0, 277), (145.0, 282),...
                   (145.0, 287), (145.0, 292), (145.0, 297), (145.0, 302), (145.0, 307), (145.0, 312)]
    for i in range(len(test_array)-1):
        x.append(test_array[i][0])
        y.append(test_array[i][1])
        print(x,y)
    # x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2, 2, height))
    z = 0.5 * np.sin(2 * x - count / 10.0) * np.sin(2 * y)
    points = np.array([x, y, z, z]).reshape(4, -1).T

    pc2 = point_cloud2.create_cloud(header, fields, points)
    pub.publish(pc2)


if __name__ == "__main__":
    rospy.init_node("pc2_publisher")
    pub = rospy.Publisher("points2", PointCloud2, queue_size=100)
    rate = rospy.Rate(10)

    count = 0
    while not rospy.is_shutdown():
        publishPC2(count)
        count += 1
        rate.sleep()