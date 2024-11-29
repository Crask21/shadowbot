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



#Test cube definition
# aruco_marker_side_length = 0.015
# aruco_marker0_pose = [-(0.025+0.0001), 0, 0, 0, -np.pi/2, 0]
# aruco_marker1_pose = [0, 0, (0.025+0.0001), 0, 0, 0]
# T_aruco_marker0 = p2T(aruco_marker0_pose)
# T_aruco_marker1 = p2T(aruco_marker1_pose)
# obj_name = "box_with_aruco"

#3D printed part definition
aruco_marker_side_length = 0.00989
aruco_marker0_pose = [0.0089378, 0.010202, (0.02+0.0001), 0, 0, -0.68964]
aruco_marker1_pose = [-0.0096386, -0.0076465, (0.02+0.0001), 0, 0, -0.81242]
T_aruco_marker0 = p2T(aruco_marker0_pose)
T_aruco_marker1 = p2T(aruco_marker1_pose)
obj_name = "part_aruco"
file_name = 'part_two_aruco_markers'

#Camera pose relative to the world frame
pose = [0, 0, 0, 0, 0, 0]

global mtx, dst, H, q_list, p_list



def get_camera_pose():
    T = p2T(pose)
    return T

def get_camera_info():
    # Get the camera matrix and distortion coefficients
    # Set the global variables mtx and dst
    global mtx, dst
    camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
    mtx = np.array(camera_info.K).reshape(3, 3)
    dst = np.array(camera_info.D)
    return mtx, dst

def get_object_pose(name):
    # Get the pose of the object with the given name
    # Returns the pose as a list [x, y, z, qx, qy, qz, qw]
    model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates)
    object_idx = model_states.name.index(name)
    object_pose = model_states.pose[object_idx]
    object_position = object_pose.position
    object_orientation = object_pose.orientation
    object_pose = [object_position.x, object_position.y, object_position.z, object_orientation.x, object_orientation.y, object_orientation.z, object_orientation.w]
    return object_pose

def MarkerPose(rvec, tvec, H, aruco_marker_T):
    R, _ = cv2.Rodrigues(rvec)
    T = np.hstack((R, tvec.reshape(3, 1)))
    T = np.vstack((T, [0, 0, 0, 1]))
    T_obj_camera_frame = np.linalg.multi_dot([T, np.linalg.inv(aruco_marker_T)])
    T_obj = np.linalg.multi_dot([H, T_obj_camera_frame])

    return T_obj, T_obj_camera_frame

def drawAxis(image, T, mtx, dst, length):
    tvec = T[:3, 3]
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    cv2.aruco.drawAxis(image, mtx, dst, rvec, tvec, length)

def pose_estimator(image_msg):
    global mtx, dst, H, q_list, p_list
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    if len(corners) > 0:
        ids = ids.flatten()
        rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_side_length, mtx, dst)

        # Draw the axis on the image
        for i in range(len(ids)):
            cv2.aruco.drawAxis(image, mtx, dst, rvecs[i], tvecs[i], aruco_marker_side_length)

        T_obj = []
        T_obj_camera_frame = []
        # Get a list of transformation matrices for each marker
        if 0 in ids and 1 in ids:
            # Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H, T_aruco_marker0)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)

            # Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H, T_aruco_marker1)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
        elif 0 in ids:
            # Only Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H, T_aruco_marker0)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)
            # print("only marker 0")
        elif 1 in ids:
            # Only Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H, T_aruco_marker1)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
            # print("only marker 1")


        # Draw the axis on the image
        for T in T_obj_camera_frame:
            drawAxis(image, T, mtx, dst, aruco_marker_side_length)
        
        T_avg = np.mean(T_obj, axis=0)

        p_avg = T_avg[:3, 3]

        # print("Error in position: {}".format(p_error))
        q_avg = tf_trans.quaternion_from_matrix(T_avg)
        print("p: ", np.around(p_avg, 4), "\tq: ", np.around(tf_trans.euler_from_quaternion(q_avg),4))


    else:
        print("No markers detected")
    cv2.imshow("Image", image)
    cv2.waitKey(10)
    # q_list.append(q_error)
    # p_list.append(p_error)


def pose_estimatortest(image_msg):
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if len(corners) > 0:
        ids = ids.flatten()
        rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_side_length, mtx, dst)

        # Draw the axis on the image
        for i in range(len(ids)):
            cv2.aruco.drawAxis(image, mtx, dst, rvecs[i], tvecs[i], aruco_marker_side_length)
    else:
        print("No markers detected")
    cv2.imshow("Image", image)
    cv2.waitKey(10)

    

def pose_estimator_node():
    rospy.init_node('pose_estimator', anonymous=True)
    r = rospy.Rate(30)
    global mtx, dst, H, q_list, p_list
    mtx, dst = get_camera_info()
    H = get_camera_pose() #Camera extrinsic matrix
    p_list, q_list = [], []
    rospy.Subscriber("/camera/color/image_raw", Image, pose_estimator)
    rospy.spin()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    pose_estimator_node()
