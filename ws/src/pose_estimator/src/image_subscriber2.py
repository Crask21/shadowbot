#!/usr/bin/env python  
import rospy

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tf_trans
from p2T import p2T
import csv
import matplotlib.pyplot as plt




def pose_estimator():
    image_msg = rospy.wait_for_message("/camera/rgb/image_raw", Image, timeout=0.1)
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.waitKey(10)

def pose_estimator_node():
    rospy.init_node('pose_estimator', anonymous=True)

    # while not rospy.is_shutdown():
    for i in range(1000):
        pose_estimator()
        if i % 100 == 0:
            print("Iteration: {}".format(i))
    
    plt.show()

    cv2.destroyAllWindows()



if __name__ == '__main__':
    pose_estimator_node()
