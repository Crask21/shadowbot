#!/usr/bin/env python  
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
import numpy as np



def spin():
    rospy.init_node('object_spin')
    pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

    modelstate = ModelState()
    modelstate.model_name = 'part_aruco'
    modelstate.pose.position.x = 1
    t0 = rospy.Time.now()
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        t = rospy.Time.now()
        dt = (t - t0).to_sec()
        w0 = 2.5
        # rotation0 = (w0 * dt % (2*np.pi))
        rotation0 = np.sin(w0 * dt)*np.pi/4
        modelstate.pose.orientation.z = np.sin(rotation0/2)
        modelstate.pose.orientation.w = np.cos(rotation0/2)

        w1 = 0.5
        r1 = 0.02
        rotation1 = w1 * dt % (2 * np.pi)
        modelstate.pose.position.y = r1 * np.sin(rotation1)
        modelstate.pose.position.x = r1 * np.cos(rotation1)

        rospy.loginfo('z: {}'.format(modelstate.pose.orientation.z))
        pub.publish(modelstate)
        print('Published')
        rate.sleep()


if __name__ == '__main__':
    try:
        spin()
    except rospy.ROSInterruptException:
        pass
        
    
    