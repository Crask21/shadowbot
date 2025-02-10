#!/usr/bin/env python  
import rospy
from geometry_msgs.msg import PoseStamped
import rtde_receive, rtde_control
import numpy as np
from tf.transformations import quaternion_matrix
import tf.transformations as tf

global pose
pose = PoseStamped
def callback(data):
    global pose
    pose = data

def demo():
    rospy.init_node('demo_node', anonymous=True)
    # rospy.Subscriber('/estimated_pose', PoseStamped, callback)
    r = rospy.Rate(5)

    rtde_c = rtde_control.RTDEControlInterface("192.168.1.130") #rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP
    rtde_c.setTcp([0, 0, 0, 0, 0, 0])
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.130")

    input("Move so object is in camera view")
    rtde_c.moveL([-0.293, -0.624, 0.386, 0.247, 3.375, 0.576], 0.2, 0.2)

    input("Receive obj pose through camera")
    pose = rospy.wait_for_message('/estimated_pose', PoseStamped)
    
    p = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
    o = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
    # Create transformation matrix from position and orientation
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = p
    transformation_matrix[:3, :3] = quaternion_matrix(o)[:3, :3]

    print("Transformation matrix: \n", transformation_matrix)
    print("Object position: ", p)
    print("Object orientation: ", o)
    print("transformation_matrix: \n", transformation_matrix)   

    list_ = rtde_r.getActualTCPPose()
    tcp_position = list_[:3]
    axis_angle = [list_[3:]/np.linalg.norm(list_[3:]), np.linalg.norm(list_[3:])]
    tcp_orientation = tf.quaternion_about_axis(axis_angle[1], axis_angle[0])
    tcp_transformation_matrix = np.eye(4)
    tcp_transformation_matrix[:3, 3] = tcp_position
    tcp_transformation_matrix[:3, :3] = quaternion_matrix(tcp_orientation)[:3, :3]

    print("TCP Transformation matrix: \n", tcp_transformation_matrix)

    # Calculate the inverse of the TCP transformation matrix
    tcp_transformation_matrix_inv = np.linalg.inv(tcp_transformation_matrix)

    # Calculate the object transformation in relation to the TCP
    obj_transformation_in_tcp = np.dot(tcp_transformation_matrix_inv, transformation_matrix)

    print("Object transformation in relation to TCP: \n", obj_transformation_in_tcp)

    # goal = [-0.150, -0.800, 0.470, 0, 0, 0]
    # goal2 = [-0.150, -0.800, 0.470, 0, np.pi, 0]
    goal = [0.098, -0.777, 0.055, 0, 0, 0]
    #rtde_c.moveL(goal2, 0.2, 0.2)

    input("Move obj to new TCP pose")
    axang = tf.rotation_from_matrix(obj_transformation_in_tcp)
    axang = axang[1]*axang[0]
    # print("axis angle: ", axang)
    new_TCP = list(obj_transformation_in_tcp[:3, 3]) + list(axang)
    # print("axis angle: ", axang)
    # print("obj pose: ", obj_transformation_in_tcp[:3, 3])
    print(new_TCP)
    print("Setting TCP")
    rtde_c.setTcp(new_TCP)
    goal2 = list(np.array(goal) + np.array([0, 0, 0.1, 0, 0, 0]))
    print("Executing movement to: ", goal2)
    rtde_c.moveL(goal2,0.1, 0.1)
    input("insert")
    rtde_c.moveL(goal,0.05, 0.05)
    s = input("Are you sure? y/n")
    if s != "n":
        rtde_c.moveL(list(np.array(goal) + np.array([0, 0, -0.0215, 0, 0, 0])),0.05, 0.05)

    input("moving out of hole again")
    rtde_c.moveL(goal2,0.1, 0.1)


    input("move to abitrary position")

    rtde_c.setTcp([0, 0, 0, 0, 0, 0])
    rtde_c.moveL([-0.1, -0.6, 0.3, 1.5,-1.5, 0.1], 0.2, 0.2)
    print("finished")
 


    
    

if __name__ == '__main__':
    demo()