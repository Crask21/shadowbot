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
from p2T import p2T



aruco_marker_side_length = 0.4


#Aruco marker pose relative to attached object 
aruco_marker0_pose = [-0.25, 0, 0, 0, -np.pi/2, 0]
aruco_marker1_pose = [0, 0, 0.25, 0, 0, 0]
obj_name = "box_with_aruco"
import tf.transformations as tf_trans

#Part definition
aruco_marker_side_length = 0.010
aruco_marker0_pose = [0.0098, 0.0098, (0.02+0.0001), 0, 0, -0.785]
aruco_marker1_pose = [-0.0098, -0.0098, (0.02+0.0001), 0, 0, -0.785]
T_aruco_marker0 = p2T(aruco_marker0_pose)
T_aruco_marker1 = p2T(aruco_marker1_pose)
obj_name = "part_aruco"
file_name = 'part_two_aruco_markers'

def get_camera_pose():
    model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates)
    camera_idx = model_states.name.index("rgbd_camera")
    camera_pose = model_states.pose[camera_idx]
    camera_position = camera_pose.position
    camera_orientation = camera_pose.orientation

    #Rotate the camera's frame of reference by 90 degrees around the y-axis due to the camera's orientation in the simulation
    
    rot_y = tf_trans.quaternion_from_euler(-np.pi/2,0,-np.pi/2)

    camera_orientation_array = tf_trans.quaternion_multiply([camera_orientation.x, camera_orientation.y, camera_orientation.z, camera_orientation.w], rot_y)
    
    camera_position_array = np.array([camera_position.x, camera_position.y, camera_position.z])
    camera_transformation_matrix = tf_trans.quaternion_matrix(camera_orientation_array)
    camera_transformation_matrix[:3, 3] = camera_position_array

    return camera_transformation_matrix
    


def pose_to_transformation_matrix(pose):

    translation = tf_trans.translation_matrix(pose[:3])
    rotation = tf_trans.euler_matrix(pose[3], pose[4], pose[5])
    transformation_matrix = tf_trans.concatenate_matrices(translation, rotation)
    return transformation_matrix

aruco_marker0_transformation_matrix = pose_to_transformation_matrix(aruco_marker0_pose)
aruco_marker1_transformation_matrix = pose_to_transformation_matrix(aruco_marker1_pose)



def get_camera_info():
    # Get the camera matrix and distortion coefficients
    # Set the global variables mtx and dst
    global mtx, dst
    camera_info = rospy.wait_for_message("/camera/rgb/camera_info", CameraInfo)
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

def pose_estimator():
    rospy.init_node('pose_estimator', anonymous=True)
    mtx, dst = get_camera_info()
    H = get_camera_pose() #Camera extrinsic matrix
    image_msg = rospy.wait_for_message("/camera/rgb/image_raw", Image, timeout=0.1)

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
            #print("ID: {}".format(ids[i]))
            #print("rvecs: {}".format(rvecs[i]))
            #print("tvecs: {}".format(tvecs[i]))
            #print("obj_points: {}".format(obj_points[i]))

        T_obj = []
        T_obj_camera_frame = []
        # Get a list of transformation matrices for each marker
        if 0 in ids and 1 in ids:
            # Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H, aruco_marker0_transformation_matrix)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)

            # Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H, aruco_marker1_transformation_matrix)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
        elif 0 in ids:
            # Only Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H, aruco_marker0_transformation_matrix)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)
        elif 1 in ids:
            # Only Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H, aruco_marker1_transformation_matrix)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)


        # Draw the axis on the image
        for T in T_obj_camera_frame:
            drawAxis(image, T, mtx, dst, aruco_marker_side_length)
        
        T_avg = np.mean(T_obj, axis=0)
        T_ground_truth = get_object_pose(obj_name)

        p_avg = T_avg[:3, 3]
        p_ground_truth = T_ground_truth[:3]
        print("Error in position: \t{}".format(p_avg - p_ground_truth))
        print("pose: \t\t\t{}".format(p_avg))
        print("ground truth: \t\t{}".format(p_ground_truth))

        q_avg = tf_trans.quaternion_from_matrix(T_avg)
        q_ground_truth = [T_ground_truth[3], T_ground_truth[4], T_ground_truth[5], T_ground_truth[6]]
        q_diff  = tf_trans.quaternion_multiply(q_avg, tf_trans.quaternion_inverse(q_ground_truth))

        print("Error in orientation (Euler): {}".format(tf_trans.euler_from_quaternion(q_diff)))






    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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



if __name__ == '__main__':
    pose_estimator()
