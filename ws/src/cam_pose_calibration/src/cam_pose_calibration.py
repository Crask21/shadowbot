#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
from cv_bridge import CvBridge
import cv2
import os
import rtde_receive




class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image1 = None
        self.image2 = None
        self.image_count = 0
        self.max_images = 20
        self.save_path = "/home/casper/Documents/studentermedhjælper/shadowbot/ws/src/cam_pose_calibration/images"
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.130")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        rospy.init_node('image_saver', anonymous=True)
        rospy.Subscriber('/cam1/color/image_raw', Image, self.callback_camera1)
        rospy.Subscriber('/cam2/color/image_raw', Image, self.callback_camera2)
        self.save_images()

    def callback_camera1(self, msg):
        self.image1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def callback_camera2(self, msg):
        self.image2 = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def save_images(self):
        with open(os.path.join(self.save_path, "pose_data.csv"), "w") as file:
            file.write("image_number,TCP,joint_positions\n")
        print("Saving images")
        for i in range(self.max_images):
            input("Press Enter to capture the next image...")
            actual_q = self.rtde_r.getActualQ()
            actual_p = self.rtde_r.getActualTCPPose()
            cv2.imwrite(os.path.join(os.path.join(self.save_path, "camera1"), f"{i}.png"), self.image1)
            # cv2.imwrite(os.path.join(os.path.join(self.save_path, "camera2"), f"{i}.png"), self.image2)
            with open(os.path.join(self.save_path, "pose_data.csv"), "a") as file:
                file.write(f"{i},{actual_p},{actual_q}\n")

            rospy.sleep(1)  # Adjust sleep time if necessary
            self.image_count += 1
            rospy.loginfo(f"Saved image pair {self.image_count}")
            print(actual_p)
        print("Done saving images")
        return EmptyResponse()

if __name__ == '__main__':
    try:
        image_saver = ImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass