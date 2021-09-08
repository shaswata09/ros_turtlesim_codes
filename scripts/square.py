#!/usr/bin/env python
import rospy
import numpy
from math import pi
from geometry_msgs.msg import Twist

rospy.init_node('square', anonymous=False)
pub = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=10)
rate = rospy.Rate(10) # 10hz

print('Starting square . . .')

# TODO: Modify the code below so that the robot moves in a square

''' Net ID: sm3843 
    Name: SHASWATA MITRA 
    Lab 1 Assisgnment Problem 1 - Make a Square'''

sideLength = 3.0
speed = 0.1
angle = pi/2
twist = Twist()

def moveLinear():
    print("Move straight...")
    twist.linear.x = 0.5
    twist.angular.z = 0
    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    current_distance = 0

    #Loop to move the turtle in an specified distance
    while(current_distance <= sideLength):
        #Publish the velocity
        pub.publish(twist)
        #Takes actual time to velocity calculus
        t1=rospy.Time.now().to_sec()
        #Calculates distancePoseStamped
        current_distance= (t1-t0)
        
    #After the loop, stops the robot
    twist.linear.x = 0
    #Force the robot to stop
    pub.publish(twist)
    rate.sleep() #sleep until the next time to publish

def rotate():
    print("Rotating 90 degrees...")
    current_angle = 0
    twist.angular.x = 0
    angular_speed = speed
    twist.angular.z = angular_speed
    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    
    while (current_angle <= angle):
        pub.publish(twist)
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed*(t1-t0)

    # Forcing our robot to stop
    twist.angular.z = 0
    pub.publish(twist)
    rate.sleep() #sleep until the next time to publish

while not rospy.is_shutdown(): 
    moveLinear()
    rotate()