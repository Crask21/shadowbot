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
import rtde_receive
import os



#Test cube definition
# aruco_marker_side_length = 0.015
# aruco_marker0_pose = [-(0.025+0.0001), 0, 0, 0, -np.pi/2, 0]
# aruco_marker1_pose = [0, 0, (0.025+0.0001), 0, 0, 0]
# T_aruco_marker0 = p2T(aruco_marker0_pose)
# T_aruco_marker1 = p2T(aruco_marker1_pose)
# obj_name = "box_with_aruco"

#3D printed part definition
# aruco_marker_side_length = 0.00989
# aruco_marker0_pose = [0.0089378, 0.010202, (0.02+0.0001), 0, 0, -0.68964]
# aruco_marker1_pose = [-0.0096386, -0.0076465, (0.02+0.0001), 0, 0, -0.81242]
# T_aruco_marker0 = p2T(aruco_marker0_pose)
# T_aruco_marker1 = p2T(aruco_marker1_pose)
obj_name = "part_aruco"
file_name = 'part_two_aruco_markers'

#Camera pose relative to the world frame
pose1 = [0, 0, 0, 0, 0, 0]
pose2 = [0, 0, 0, 0, 0, 0]

# Aruco marker pose for flange
T_aruco_marker0 = p2T([-0.033349,-0.035891,0.0062,0,0,-1.5552/180*np.pi]) #Missing correction for 6.2mm in z direction
T_aruco_marker1 = p2T([0.033841,0.036006,0.0062,0,0,-2.9169/180*np.pi])

# Aruco marker pose for demo part
# T_aruco_marker0 = p2T([-0.0078981, 0.012777, 0.025, 0, 0, -3.2298/180*np.pi])
# T_aruco_marker1 = p2T([0.0057998, -0.012272, 0.025, 0, 0, -1.8049/180*np.pi])

aruco_marker_side_length = 0.0203 # Real world marker size
# T1 = [[-0.28587316407227642, -0.51317057736526661, 0.80927899552002736, -0.5317133663644176],
#       [-0.93384342790334807, -0.040266690318529585, -0.35540828045012723, -0.3628481841155831],
#       [0.21497205917288395, -0.85734156097636527, -0.46770980489784414, 0.4461471911440853], 
#       [0, 0, 0, 1]] # Old calculation of position of cam1 on pillar
T1 = [[-0.28080641108408705, -0.52529853656550363, 0.80324915622502535, -0.5340115894520676],
   [-0.92574746104020345, -0.072599121646612613, -0.37110780901748408, -0.3620663612649232],
   [0.25325757219023431, -0.84781531893348128, -0.4659074877162071, 0.4409975675038182],
   [0, 0, 0, 1]] # New calculation of position of cam1 on pillar
# T1 = np.array(T1)
# T1[:3, :3] = T1[:3, :3].T
# T1 = T1.tolist()
# print(T1)
# T2 = [[-0.27950358555639854, 0.52866365701832529, 0.8014939072813515, -0.5282833629794542],
#       [0.93678722651385238, -0.032852793954910961, 0.3483538232456192, -0.8076321694211532], 
#       [0.2104933203253751, 0.84819539710927838, -0.48606288731092429,0.4422647013629318],
#       [0, 0, 0, 1]] # Old calculation of position of cam2 on pillar
T2 = [[-0.28549835357294084, 0.55042222520351669, 0.78455469159845648, -0.5286516502211807],
   [0.93083198695487401, -0.035606064332653592, 0.36370870245896036, -0.8068283958501908],
   [0.22812825815488452, 0.8341268381875433, -0.50218514080612753, 0.4407541458165125],
   [0, 0, 0, 1]] # New calculation of position of cam2 on pillar
# Inverse the rotation matrix in T2
# T2 = np.array(T2)
# T2[:3, :3] = T2[:3, :3].T
# T2 = T2.tolist()
# print(T2)
global mtx, dst, H1, q_list, p_list
global mtx2, dst2, H2
global T1_avg, T2_avg
# T2_avg = None



def get_camera_pose(pose):
    T = p2T(pose)
    return T

def get_cam1_info():
    # Get the camera matrix and distortion coefficients
    # Set the global variables mtx and dst
    global mtx, dst
    camera_info = rospy.wait_for_message("/cam1/color/camera_info", CameraInfo)
    mtx = np.array(camera_info.K).reshape(3, 3)
    # dst = np.array(camera_info.D)
    # dst = np.array([0.05746940315041734,0.27530099251372153,0.0006999800812428493,-0.000768091966152859,-1.3372741520547178]) #640x480
    dst = np.array([0.139874415251927, -0.42450290350724, -0.000443565714817, -0.0013455338084, 0.31016064591688]) # k1, k2, p1, p2, k3 #1920x1080

    return mtx, dst
def get_cam2_info():
    # Get the camera matrix and distortion coefficients
    # Set the global variables mtx and dst
    global mtx2, dst2
    camera_info = rospy.wait_for_message("/cam2/color/camera_info", CameraInfo)
    mtx2 = np.array(camera_info.K).reshape(3, 3)
    # dst = np.array(camera_info.D)
    # dst2 = np.array([0.08100698078726337,0.11095057485263705,0.0027741518864256662,-0.0008486478641119692,-0.9360584424119222])
    dst2 = np.array([0.13307403966671, -0.331350606935734, 0.000245071560936, 0.000315038110404, 0.123296864151717]) # k1, k2, p1, p2, k3 #1920x1080
    return mtx2, dst2

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
    cv2.drawFrameAxes(image, mtx, dst, rvec, tvec, length)


def pose_estimator1(image_msg):
    global mtx, dst, H1, q_list, p_list
    global T1_avg
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = detector.detectMarkers(image)

    # 3D points of the aruco marker
    obj_points = np.array([
            [-aruco_marker_side_length / 2, aruco_marker_side_length / 2, 0],   # Top-left corner
            [aruco_marker_side_length / 2, aruco_marker_side_length / 2, 0],    # Top-right corner
            [aruco_marker_side_length / 2, -aruco_marker_side_length / 2, 0],   # Bottom-right corner
            [-aruco_marker_side_length / 2, -aruco_marker_side_length / 2, 0]   # Bottom-left corner
        ], dtype=np.float32)

    # print("corners: ", corners)
    # print("Len corners: ", len(corners))
    # print("ids: ", ids)
    if len(corners) > 0:
        ids = ids.flatten()
        rvecs = []
        tvecs = []
        for i in range(len(ids)):
            # rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(corners[i], aruco_marker_side_length, mtx, dst)
            (retval, rvec, tvec) = cv2.solvePnP(obj_points, corners[i], mtx, dst)
            rvecs.append(rvec)
            tvecs.append(tvec)
        # (retval, rvecs, tvecs) = cv2.solvePnP(obj_points, corners[0], mtx, dst)


        # Draw the axis on the image
        # print("rvecs: ", rvecs)
        # print("tvecs: ", tvecs)
        for i in range(len(ids)):
            cv2.drawFrameAxes(image, mtx, dst, rvecs[i], tvecs[i], aruco_marker_side_length)
        
        # --------------------------------- Debugging -------------------------------- #
        # print("rvecs: ", rvecs)
        # print("tvecs: ", tvecs)
        # print("H1: ", H1)
        # print("T_aruco_marker0: ", T_aruco_marker0)

        T_obj = []
        T_obj_camera_frame = []
        # Get a list of transformation matrices for each marker
        if 0 in ids and 1 in ids:
            # Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H1, T_aruco_marker0)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)

            # Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H1, T_aruco_marker1)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
        elif 0 in ids:
            # Only Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H1, T_aruco_marker0)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)
            # print("only marker 0")
        elif 1 in ids:
            # Only Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H1, T_aruco_marker1)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
            # print("only marker 1")


        # Draw the axis on the image
        for T in T_obj_camera_frame:
            drawAxis(image, T, mtx, dst, aruco_marker_side_length)
        # drawAxis(image, T_obj_camera_frame[0], mtx, dst, aruco_marker_side_length)
        
        # print("T_obj: ", T_obj)
        T_avg = np.mean(T_obj, axis=0)
        # print("T_avg: ", T_avg)

        global T1_avg
        alpha = 0.5  # Smoothing factor for the low-pass filter
        if T1_avg is not None:
            T1_avg = alpha * T_avg + (1 - alpha) * T1_avg
        else:
            T1_avg = T_avg

        p_avg = T_avg[:3, 3]

        # print("Error in position: {}".format(p_error))
        q_avg = tf_trans.quaternion_from_matrix(T_avg)
        # print("p1: ", np.around(p_avg, 4), "\tq1: ", np.around(tf_trans.euler_from_quaternion(q_avg),4))
        temp = np.array(T_obj_camera_frame[0])[0:3, 3]
        # print("T_cam_1: ", T_obj_camera_frame)

        # print("p_avg: ", p_avg)
    else:
        T1_avg = None
        print("Cam1, No markers detected")
    # global T2_avg
    if T2_avg is not None and not np.isnan(T2_avg).any():
        T2_cam1 = np.linalg.multi_dot([np.linalg.inv(H1), T2_avg])
        drawAxis(image, T2_cam1, mtx, dst, aruco_marker_side_length)
        # print("T2_cam1: ", T2_cam1)
        cv2.putText(image, "T2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Image 1", image)
    cv2.waitKey(10)
    # q_list.append(q_error)
    # p_list.append(p_error)

def pose_estimator2(image_msg):
    global mtx2, dst2, H2, q2_list, p2_list
    global T2_avg
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = detector.detectMarkers(image)
    # 3D points of the aruco marker
    obj_points = np.array([
            [-aruco_marker_side_length / 2, aruco_marker_side_length / 2, 0],   # Top-left corner
            [aruco_marker_side_length / 2, aruco_marker_side_length / 2, 0],    # Top-right corner
            [aruco_marker_side_length / 2, -aruco_marker_side_length / 2, 0],   # Bottom-right corner
            [-aruco_marker_side_length / 2, -aruco_marker_side_length / 2, 0]   # Bottom-left corner
        ], dtype=np.float32)

    # print("corners: ", corners)
    # print("Len corners: ", len(corners))
    # print("ids: ", ids)
    if len(corners) > 0:
        ids = ids.flatten()
        rvecs = []
        tvecs = []
        for i in range(len(ids)):
            # rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(corners[i], aruco_marker_side_length, mtx, dst)
            (retval, rvec, tvec) = cv2.solvePnP(obj_points, corners[i], mtx, dst)
            rvecs.append(rvec)
            tvecs.append(tvec)
        

        # Draw the axis on the image
        for i in range(len(ids)):
            cv2.drawFrameAxes(image, mtx2, dst2, rvecs[i], tvecs[i], aruco_marker_side_length)
           
        T_obj = []
        T_obj_camera_frame = []
        # Get a list of transformation matrices for each marker

        



        if 0 in ids and 1 in ids:
            # Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H2, T_aruco_marker0)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)

            # Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H2, T_aruco_marker1)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
        elif 0 in ids:
            # Only Marker 0
            idx = np.where(ids == 0)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj0_T, obj0_T_Camera = MarkerPose(rvec, tvec, H2, T_aruco_marker0)
            T_obj.append(obj0_T)
            T_obj_camera_frame.append(obj0_T_Camera)
            # print("only marker 0")
        elif 1 in ids:
            # Only Marker 1
            idx = np.where(ids == 1)[0][0]
            rvec, tvec = rvecs[idx], tvecs[idx]
            obj1_T, obj1_T_Camera = MarkerPose(rvec, tvec, H2, T_aruco_marker1)
            T_obj.append(obj1_T)
            T_obj_camera_frame.append(obj1_T_Camera)
            # print("only marker 1")

        if 0 in ids or 1 in ids:
            # Draw the axis on the image
            for T in T_obj_camera_frame:
                drawAxis(image, T, mtx2, dst2, aruco_marker_side_length)
            
            T_avg = np.mean(T_obj, axis=0)
            # print("T_obj: ", T_obj)
            # print("T_avg: ", T_avg)

            global T2_avg
            alpha = 0.5  # Smoothing factor for the low-pass filter
            if T2_avg is not None:
                T2_avg = alpha * T_avg + (1 - alpha) * T2_avg
            else:
                T2_avg = T_avg

            p_avg = T_avg[:3, 3]

            # print("Error in position: {}".format(p_error))
            q_avg = tf_trans.quaternion_from_matrix(T_avg)
            # print("p2: ", np.around(p_avg, 4), "\tq1: ", np.around(tf_trans.euler_from_quaternion(q_avg),4))


            # print("T_avg: ", T_avg)
    else:
        T2_avg = None
        print("Cam2, No markers detected")
    # cv2.imshow("Image 2", image)
    # cv2.waitKey(10)
    # q_list.append(q_error)
    # p_list.append(p_error)

    

def pose_estimator_node():
    print("-------------------------------------------\n\n start \n\n-------------------------------------------")
    rospy.init_node('pose_estimator', anonymous=True)
    print("-------------------------------------------\n\n node created\n\n-------------------------------------------")
    pose_pub = rospy.Publisher('estimated_pose', PoseStamped, queue_size=10)

    r = rospy.Rate(30)
    global mtx, dst, H1, q_list, p_list, H2
    mtx, dst = get_cam1_info()
    H1 = T1
    H2 = T2
    p_list, q_list = [], []

    global mtx2, dst2
    mtx2, dst2 = get_cam2_info()
    global T1_avg, T2_avg
    T2_avg = None
    T1_avg = None



    # ---------------------------------- UR Arm ---------------------------------- #
    use_arm = False
    if use_arm:
        rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.130")

    print("-------------------------------------------\n\n\n-------------------------------------------")
    rospy.Subscriber("/cam1/color/image_raw", Image, pose_estimator1)
    rospy.Subscriber("/cam2/color/image_raw", Image, pose_estimator2)
    print("-------------------------------------------\n\n initialised \n\n-------------------------------------------")
    save_path = "/home/casper/Documents/studentermedhjælper/shadowbot/ws/src/pose_estimator/data/real_error_new_cam_pose.csv"
    # save_path_Cam1 = "/home/casper/Documents/studentermedhjælper/shadowbot/ws/src/pose_estimator/data/real_error_new_cam_pose.csv"
    # save_path_Cam2 = "/home/casper/Documents/studentermedhjælper/shadowbot/ws/src/pose_estimator/data/real_error_new_cam_pose.csv"
    
    if use_arm:
        with open(save_path, "w") as file:
            file.write("p1, q1, p2, q2, p_actual\n")
    i = 0
    while not rospy.is_shutdown():
        r.sleep()


        if T1_avg is not None and T2_avg is not None and not np.isnan(T2_avg).any() and not np.isnan(T1_avg).any():
            T_avg = np.mean([T1_avg, T2_avg], axis=0)
            

            p1 = T1_avg[:3, 3]
            p2 = T2_avg[:3, 3]
            p_mid = (p1 + p2) / 2
            q1 = tf_trans.quaternion_from_matrix(T1_avg)
            q2 = tf_trans.quaternion_from_matrix(T2_avg)
            q_mid = tf_trans.quaternion_slerp(q1, q2, 0.5)
            p_diff = np.linalg.norm(p1 - p2)
            q_diff = tf_trans.quaternion_multiply(q1, tf_trans.quaternion_inverse(q2))
            
            if use_arm:
                actual_q = rtde_r.getActualTCPPose()
                actual_p = actual_q[:3]
                #actual_q_quat = tf_trans.quaternion_(actual_q[3], actual_q[4], actual_q[5])
                # q_error = tf_trans.quaternion_multiply(q_mid, tf_trans.quaternion_inverse(actual_q_quat))
                p_error = p_mid - actual_p
                # print("Error in position: {}, \t\tError in orientation: {}".format(p_error, tf_trans.euler_from_quaternion(q_error)))
                print(i)
                p1 = p1.tolist()
                p2 = p2.tolist()
                q1 = q1.tolist()
                q2 = q2.tolist()
                # actual_q_quat = actual_q_quat.tolist()
                # print("p1 type: {}, q1 type: {}, actual_p type: {}".format(type(p1), type(q1), type(actual_p)))
                # print("p1: {}, \tq1: {}, \tactual_q: {}".format(np.around(p1, 4), np.around(q1,4), actual_q))
                i += 1
                with open(save_path, "a") as file:
                    file.write(", ".join(map(str, p1 + q1 + p2 + q2 + actual_p + actual_q[3:])) + "\n")
        elif T1_avg is None:
            T_avg = T2_avg
        elif T2_avg is None:
            T_avg = T1_avg
        
        if T_avg is not None and not np.isnan(T_avg).any():
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "world"

            pose_msg.pose.position.x = T_avg[0, 3]
            pose_msg.pose.position.y = T_avg[1, 3]
            pose_msg.pose.position.z = T_avg[2, 3]

            q_avg = tf_trans.quaternion_from_matrix(T_avg)
            pose_msg.pose.orientation.x = q_avg[0]
            pose_msg.pose.orientation.y = q_avg[1]
            pose_msg.pose.orientation.z = q_avg[2]
            pose_msg.pose.orientation.w = q_avg[3]

            pose_pub.publish(pose_msg)
        if(i >= 1000):
            break 
            # else:
            #     print("Error in position: {}, \t\tError in orientation: {}".format(p_diff, tf_trans.euler_from_quaternion(q_diff)))
    # rospy.spin()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    pose_estimator_node()
