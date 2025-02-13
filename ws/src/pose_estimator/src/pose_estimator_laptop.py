#!/usr/bin/env python  
import rospy


import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tf_trans
from p2T import p2T
import csv
import matplotlib.pyplot as plt
import rtde_receive
import os
import yaml
import rospkg



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
# T_aruco_marker0 = p2T(aruco_marker0_pose)
# T_aruco_marker1 = p2T(aruco_marker1_pose)
obj_name = "part_aruco"
file_name = 'part_two_aruco_markers'

#Camera pose relative to the world frame
pose1 = [0, 0, 0, 0, 0, 0]
pose2 = [0, 0, 0, 0, 0, 0]

# Aruco marker pose for flange
# T_aruco_marker0 = p2T([-0.033349,-0.035891,0.0062,0,0,-1.5552/180*np.pi]) #Missing correction for 6.2mm in z direction
# T_aruco_marker1 = p2T([0.033841,0.036006,0.0062,0,0,-2.9169/180*np.pi])

# Aruco marker pose for demo part
# T_aruco_marker0 = p2T([-0.0078981, 0.012777, 0.025, 0, 0, -3.2298/180*np.pi])
# T_aruco_marker1 = p2T([0.0057998, -0.012272, 0.025, 0, 0, -1.8049/180*np.pi])

# aruco_marker_side_length = 0.0203 # Real world marker size

class ArucoObject:
    def __init__(self, aruco_marker_side_length, marker_poses: np.ndarray, marker_ids) -> None:
        # aruco_marker_side_length: Side length of the aruco marker in meters
        # marker_poses: List of marker poses in the object frame. Each row represents a marker pose in the form [x, y, z, roll, pitch, yaw]
        self.marker_length = aruco_marker_side_length
        self.marker_poses = marker_poses
        self.marker_points = self.get_3d_points(marker_poses)
        self.marker_ids = marker_ids
        self.board = cv2.aruco.Board(
            self.marker_points,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),
            self.marker_ids
        )       

    def plot_markers(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for j, points in enumerate(self.marker_points):
            for i, point in enumerate(points):
                ax.scatter(point[0], point[1], point[2])
                ax.text(point[0], point[1], point[2], f'{self.marker_ids[j]}:{i%4}', size=12, zorder=1, color='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def get_3d_points(self, marker_poses: np.ndarray) -> np.ndarray:
        # marker_poses: List of marker poses in the object frame. Each row represents a marker pose in the form [x, y, z, roll, pitch, yaw]
        # Returns the 3D points of the aruco markers in the object frame
        obj_points = []
        for pose in marker_poses:
            obj_points.append([])
            x, y, z, roll, pitch, yaw = pose
            # Calculate the 3D points of the aruco marker
            # Create a rotation matrix from the yaw angle
            R = tf_trans.euler_matrix(0, 0, yaw)[:3, :3]
            # Define the corners of the marker in its local frame
            local_corners = np.array([
                [-self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, -self.marker_length / 2, 0],
                [-self.marker_length / 2, -self.marker_length / 2, 0]
            ])
            # Rotate and translate the corners to the global frame
            global_corners = np.dot(R, local_corners.T).T + np.array([x, y, z])
            obj_points[-1].extend(global_corners.tolist())

        return np.array(obj_points, dtype=np.float32)
    
    def matchImagePoints(self, corners, ids):

        (obj_points, img_points) = self.board.matchImagePoints(corners, ids)

        return obj_points, img_points

class Camera:
    def __init__(self, name = None, H: np.ndarray = None, mtx: np.ndarray= None, dst: np.ndarray= None, filename: str = '') -> None:
        # Initialize camera object with the given name, transformation matrix, camera matrix and distortion coefficients
        if filename:
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
                self.name = data['name']
                self.topic = data['topic']
                self.H = np.array(data['H']).reshape(4, 4)
                self.mtx = np.array(data['camera_matrix']).reshape(3, 3)
                self.dst = np.array(data['distortion_coefficients'])
        else:
            self.name = name
            self.topics = ["/" + name + "/color/image_raw", "/" + name + "/color/camera_info"]
            self.H = H      # Transformation matrix of the camera
            self.mtx = mtx  # Camera matrix
            self.dst = dst  # Distortion coefficients (k1, k2, p1, p2, k3)

    def __str__(self) -> str:
        return f"Camera: {self.name}\nH: {self.H}\nCamera matrix: {self.mtx}\nDistortion coefficients: {self.dst}"
    
    def get_marker_poses(self, corners, ids, obj: ArucoObject):
        # corners: List of detected corners of the aruco markers
        # ids: List of detected aruco marker ids
        # obj: ArucoObject object
        # Returns the 3D points of the aruco markers in the object frame and the 2D points in the image frame
        if not any(id in obj.marker_ids for id in ids):
            return None, None, False
        (obj_points, img_points) = obj.matchImagePoints(corners, ids)
        
        (retval, rvecs, tvecs) = cv2.solvePnP(obj_points, img_points, self.mtx, self.dst)
        
        return retval, rvecs, tvecs


        



def main():

    # ---------------------------- initialise Cameras ---------------------------- #
    mtx = np.array([[9.067430417914442e+02,0,6.547558246691825e+02],[0,9.068560175984609e+02,3.839032451175279e+02],[0,0,1]])
    rdst = [0.054336313944072,-0.393173585952356,0.507353318581503]
    tdst = [-0.001249715127214,-0.001492447730050]
    dst = np.array([rdst[0], rdst[1], tdst[0], tdst[1], rdst[2]])
    H = np.eye(4)
    cam1 = Camera("camera1", H, mtx, dst)

    obj = ArucoObject(aruco_marker_side_length, np.array([aruco_marker0_pose, aruco_marker1_pose]), np.array([0, 1], dtype=np.int32))

    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    while True:
        ret, frame = cam.read()

        
        (corners, ids, rejected) = detector.detectMarkers(frame)
        if len(corners) > 0:
            (ret, rvec, tvec) = cam1.get_marker_poses(corners, ids, obj)
            if rvec is not None:
                print("rvec: ", rvec)
                print("tvec: ", tvec)
                # Draw the detected markers
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, mtx, dst, rvec, tvec, aruco_marker_side_length)


        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
