# CV Pose estimation for Shadowbot

## Usage Guide

The main pose estimator code is found in the pose_estimator package found in ws/src/pose_estimator/. To use the code for pose estimation, follow these steps:

1. **Start the Pose Estimator Node**:
   Run the following command to start the camera recording and publishing the `PoseStamped` message to the `/estimated_pose` topic:

   ```sh
   rosrun pose_estimator pose_estimator_cam_3.py
   ```

2. **Read and Display the Data**:
   The file `pose_estimator/Haegu_rotation_tracker_demo.py` demonstrates how the data can be read and displayed. You can refer to this file for an example implementation.

3. **Structure of Code**:
   The code is structured around two classes: ArucoObject and Camera. ArucoObject contructs an aruco board of the specified object. Camera takes a .yaml file as input that contains intrisics, distortion coefficients and more.
