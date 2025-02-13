#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import time
import math
import numpy as np
import tf.transformations as tf_trans
import matplotlib.pyplot as plt


class RotationTracker:
    def __init__(self):
        self.z_rotations = []
        self.start_time = time.time()
        self.subscriber = rospy.Subscriber('/estimated_pose', PoseStamped, self.callback)

    def callback(self, msg):
        current_time = time.time()
        if current_time - self.start_time <= 10:
            orientation = msg.pose.orientation
            z_rotation = self.quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)[2]
            self.z_rotations.append(z_rotation)
        else:
            rospy.signal_shutdown('Finished recording rotations')

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def plot_rotations(self):
        self.z_rotations = np.array(self.z_rotations) - self.z_rotations[0]
        plt.plot(self.z_rotations)
        plt.xlabel('Time (s)')
        plt.ylabel('Rotation around Z axis (radians)')
        plt.title('Change in Rotation around Z axis over Time')
        plt.show()

if __name__ == '__main__':
    rospy.init_node('rotation_tracker', anonymous=True)
    tracker = RotationTracker()
    rospy.spin()
    tracker.plot_rotations()