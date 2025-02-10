#!/usr/bin/env python  
import rospy
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tf_trans
from p2T import p2T
import csv
import matplotlib.pyplot as plt
import yaml
import rospkg
import pyrealsense2 as rs



#3D printed part definition
# aruco_marker_side_length = 0.00989
aruco_marker0_pose = [0.0089378, 0.010202, (0.02+0.0001), 0, 0, -0.68964]
aruco_marker1_pose = [-0.0096386, -0.0076465, (0.02+0.0001), 0, 0, -0.81242]
# T_aruco_marker0 = p2T(aruco_marker0_pose)
# T_aruco_marker1 = p2T(aruco_marker1_pose)

# Aruco marker pose for flange
# T_aruco_marker0 = p2T([-0.033349,-0.035891,0.0062,0,0,-1.5552/180*np.pi]) #Missing correction for 6.2mm in z direction
# T_aruco_marker1 = p2T([0.033841,0.036006,0.0062,0,0,-2.9169/180*np.pi])

# Aruco marker pose for demo part
# T_aruco_marker0 = p2T([-0.0078981, 0.012777, 0.025, 0, 0, -3.2298/180*np.pi])
# T_aruco_marker1 = p2T([0.0057998, -0.012272, 0.025, 0, 0, -1.8049/180*np.pi])

aruco_marker_side_length = 0.0203 # Real world marker size
   
class ArucoObject:
    def __init__(self, aruco_marker_side_length, marker_poses: np.ndarray, marker_ids) -> None:
        # aruco_marker_side_length: Side length of the aruco marker in meters
        # marker_poses: List of marker poses in the object frame. Each row represents a marker pose in the form [x, y, z, roll, pitch, yaw]
        # marker_ids: List of matching marker ids to the corresponding marker poses
        # Initialize the ArucoObject with the given marker side length, marker poses and marker ids
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
        # Helper function to plot the aruco markers in 3D
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
        # corners: List of detected corners of the aruco markers
        # ids: List of detected aruco marker ids
        # Returns the 3D points of the aruco markers in the object frame and the 2D points in the image frame
        (obj_points, img_points) = self.board.matchImagePoints(corners, ids)

        return obj_points, img_points

class Camera:
    def __init__(self, name = None, H: np.ndarray = None, mtx: np.ndarray= None, dst: np.ndarray= None, filename: str = '') -> None:
        # Initialize camera object with the given name, transformation matrix, camera matrix and distortion coefficients
        # Code will currently not work without a filename
        if filename:
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
                self.name = data['name']
                self.topic = data['topic']
                self.H = np.array(data['H']).reshape(4, 4)
                self.mtx = np.array(data['camera_matrix']).reshape(3, 3)
                self.dst = np.array(data['distortion_coefficients'])
                self.pipe = rs.pipeline()
                self.cfg = rs.config()
                self.cfg.enable_device(data['serial_no'])
                self.cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
                self.pipe.start(self.cfg)

                self.alpha = 0.8  # Smoothing factor for the low-pass filter
                self.rvec_est = np.zeros((3, 1))
                self.tvec_est = np.zeros((3, 1))
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
            return False, None, None
        (obj_points, img_points) = obj.matchImagePoints(corners, ids)
        
        (retval, rvecs, tvecs) = cv2.solvePnP(obj_points, img_points, self.mtx, self.dst)
        
        return retval, rvecs, tvecs

    def get_image(self):
        # Get the image from the camera
        frame = self.pipe.wait_for_frames()
        color_frame = frame.get_color_frame()

        image = np.asanyarray(color_frame.get_data())
        return image
    
    def detect_pose(self, obj: ArucoObject, show=False):
        # obj: ArucoObject object
        # show: If True, display the image with the detected markers
        # Detect the aruco markers in the image and estimate the pose of the object
        image = self.get_image()
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        (corners, ids, rejected) = detector.detectMarkers(image)


        if len(corners) > 0:
            ids = ids.flatten()
            retval, rvec, tvec = self.get_marker_poses(corners, ids, obj)
            
            if retval:
                self.rvec_est = self.alpha * rvec + (1 - self.alpha) * self.rvec_est
                self.tvec_est = self.alpha * tvec + (1 - self.alpha) * self.tvec_est
                cv2.drawFrameAxes(image, self.mtx, self.dst, self.rvec_est, self.tvec_est, obj.marker_length)
            else:
                print("No markers detected")

            image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

        
        if show:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow(self.name, image)
            cv2.waitKey(10)
    
    def T_est(self):
        # Returns the transformation matrix of the camera in the world frame
        T_cam_frame = np.eye(4)
        T_cam_frame[:3, :3] = cv2.Rodrigues(self.rvec_est)[0]
        T_cam_frame[:3, 3] = self.tvec_est.flatten()
        T = np.linalg.multi_dot([self.H, T_cam_frame])
        return T


def pose_estimator_node():
    # ------------------------------ Ros node start ------------------------------ #
    rospy.init_node('pose_estimator', anonymous=True)
    pose_pub = rospy.Publisher('estimated_pose', PoseStamped, queue_size=10)
    r = rospy.Rate(15)

    # ---------------------------- initialise Cameras ---------------------------- #
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('pose_estimator')
    config_path = f"{package_path}/config/camera1.yaml"
    cam1 = Camera(filename = config_path)

    config_path = f"{package_path}/config/camera2.yaml"
    cam2 = Camera(filename = config_path)

    # ----------------------------- initialise object ---------------------------- #
    aruco_marker_side_length = 0.00989 # Marker size
    aruco_marker0_pose = [-0.0059644, 0.019172, 0.03, 0, 0, 16.613/180*np.pi]
    aruco_marker1_pose = [0.0045943, -0.01967, 0.03, 0, 0, 13.347/180*np.pi]
    obj = ArucoObject(aruco_marker_side_length, np.array([aruco_marker0_pose, aruco_marker1_pose]), np.array([0, 1], dtype=np.int32))

    
    while not rospy.is_shutdown():
        cam1.detect_pose(obj, show=False)
        cam2.detect_pose(obj, show=True)

        T1 = cam1.T_est()
        T2 = cam2.T_est()

        T = T2

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"

        pose_msg.pose.position.x = T[0, 3]
        pose_msg.pose.position.y = T[1, 3]
        pose_msg.pose.position.z = T[2, 3]

        q = tf_trans.quaternion_from_matrix(T)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        pose_pub.publish(pose_msg)
        r.sleep()



if __name__ == '__main__':
    pose_estimator_node()
