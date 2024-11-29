import rtde_receive
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.130")
actual_q = rtde_r.getActualTCPPose()

print(actual_q)