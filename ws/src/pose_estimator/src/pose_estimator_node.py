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

def pose_estimator(mtx, dst, H):
    image_msg = rospy.wait_for_message("/camera/rgb/image_raw", Image)

    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # print("Number of markers detected: {}".format(len(corners)))
    # cv2.imshow("Image", image)
    # cv2.waitKey()
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
            print("only marker 0")
        elif 1 in ids:
            # Only Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H, T_aruco_marker1)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
            print("only marker 1")


        # Draw the axis on the image
        for T in T_obj_camera_frame:
            drawAxis(image, T, mtx, dst, aruco_marker_side_length)
        
        T_avg = np.mean(T_obj, axis=0)
        T_ground_truth = get_object_pose(obj_name)

        p_avg = T_avg[:3, 3]
        p_ground_truth = T_ground_truth[:3]
        p_error = p_avg - p_ground_truth

        # print("Error in position: {}".format(p_error))

        q_avg = tf_trans.quaternion_from_matrix(T_avg)
        q_ground_truth = [T_ground_truth[3], T_ground_truth[4], T_ground_truth[5], T_ground_truth[6]]
        q_error  = tf_trans.quaternion_multiply(q_avg, tf_trans.quaternion_inverse(q_ground_truth))

        # print("Error in orientation (Euler): {}".format(tf_trans.euler_from_quaternion(q_error)))
    else:
        print("No markers detected")
        cv2.imshow("Image", image)
        cv2.waitKey()
    if np.linalg.norm(p_error) > 0.01:
        print("Position error exceeds threshold: {}".format(np.linalg.norm(p_error)))
        print("Position error: {}".format(p_error))
        print("pose: {}".format(p_avg))
        print("ground truth: {}".format(p_ground_truth))
        cv2.imshow("Image", image)
        cv2.waitKey()
    cv2.waitKey(10)
    return p_error, q_error

def save_data(p_list, q_list, title):
    with open(f'/home/casper/Documents/studentermedhjælper/shadowbot/ws/src/pose_estimator/data/{title}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['p_error_x', 'p_error_y', 'p_error_z', 'q_error_x', 'q_error_y', 'q_error_z', 'q_error_w'])
        for p, q in zip(p_list, q_list):
            writer.writerow([p[0], p[1], p[2], q[0], q[1], q[2], q[3]])

def pose_estimator_node():
    rospy.init_node('pose_estimator', anonymous=True)
    mtx, dst = get_camera_info()
    H = get_camera_pose() #Camera extrinsic matrix

    # while not rospy.is_shutdown():
    p_list, q_list = [], []
    for i in range(1000):
        p_error, q_error = pose_estimator(mtx, dst, H)
        p_list.append(p_error)
        q_list.append(q_error)
        if i % 100 == 0:
            print("Iteration: {}".format(i))
    
    p_avg = np.mean(p_list, axis=0)
    q_avg = np.mean(q_list, axis=0)
    print("Average position error: {}".format(p_avg))
    print("Average orientation error (Euler): {}".format(tf_trans.euler_from_quaternion(q_avg)))


    
    save_data(p_list, q_list, file_name)
    # Convert p_list to a numpy array for easier manipulation
    p_array = np.array(p_list)

    # Plot histograms for each component of the position error
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Histogram for Two Aruco Markers {} samples\n'.format(len(p_list)))
    

    axs[0].hist(p_array[:, 0], bins=25, color='r', alpha=0.7)
    axs[0].set_title('Histogram of Position Error in X')
    axs[0].set_xlabel('Error (m)')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(p_array[:, 1], bins=25, color='g', alpha=0.7)
    axs[1].set_title('Histogram of Position Error in Y')
    axs[1].set_xlabel('Error (m)')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(p_array[:, 2], bins=25, color='b', alpha=0.7)
    axs[2].set_title('Histogram of Position Error in Z')
    axs[2].set_xlabel('Error (m)')
    axs[2].set_ylabel('Frequency')
    # plt.subplots_adjust(top=0.85)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust 'rect' to leave space for supertitle

    plt.show()

    cv2.destroyAllWindows()



if __name__ == '__main__':
    pose_estimator_node()
