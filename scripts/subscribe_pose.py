#! /usr/bin/env python
import rospy
from turtlesim.msg import Pose

def callback(pose):
    print(f"Pose (x,y,theta): ({pose.x:.2f},{pose.y:.2f},{pose.theta:.2f})")


rospy.init_node('subscribe_pose')
odom_sub = rospy.Subscriber('/turtle1/pose', Pose, callback)
rospy.spin()