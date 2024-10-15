#!/usr/bin/env python  
import rospy

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

def imageCallback(image):
    print("Image received")
    bridge = CvBridge()
    
    try:
        cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
        cv2.imshow("Image Window", cv_image)
        cv2.waitKey(3)
    except CvBridgeError as e:
        print(e)



def pose_estimator():
    rospy.init_node('pose_estimator', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_raw", Image, imageCallback)
    rospy.spin()

if __name__ == '__main__':
    pose_estimator()
