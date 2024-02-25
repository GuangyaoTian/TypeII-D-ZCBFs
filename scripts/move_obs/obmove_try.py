#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Point
import numpy as np
import time 
def control_obstacle_speed():
    rospy.init_node('control_obstacle_speed')

    set_model_state_obstacle1 = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    obstacle1_name = 'obstacle_1'

    rate = rospy.Rate(10)  

    while not rospy.is_shutdown():
        
        current_time = rospy.get_time()

        
        position_obstacle1 = Point()
        position_obstacle1.x = 2 + 0.2*np.sin(current_time) 
        position_obstacle1.y = 2 

        model_state_obstacle1 = ModelState()
        model_state_obstacle1.model_name = obstacle1_name
        
        model_state_obstacle1.pose.position = position_obstacle1

        
        set_model_state_obstacle1(model_state_obstacle1)

        rate.sleep()

rospy.init_node('control_obstacle_speed')

set_model_state_obstacle1 = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
###set_model_state_obstacle4 = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
###set_model_state_obstacle5 = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
orbstacle1_name = 'obstacle_1'
orbstacle4_name = 'obstacle_4'
orbstacle5_name = 'obstacle_5'
orbstacl11_name = 'obstacle_11'
###orbstacle2_name = 'obstacle_2'
rate = rospy.Rate(10)  
start_time=time.time()

while not rospy.is_shutdown():
        ###twist_obstacle1 = Twist()
        ###twist_obstacle1.linear.x = 0.2  
        
    current_time=time.time()-start_time
    position_obstacle1 = Point()
    if current_time < 46:
        position_obstacle1.x = 1.7 +0.05*current_time
        position_obstacle1.y = 2.3-0.05*current_time
    else :
        position_obstacle1.y = 0
        position_obstacle1.x = 4 
    ###position_obstacle2 = Point()
    ###position_obstacle2.x = 2.8 +0.1*np.cos(2*current_time) 
    ###position_obstacle2.y = -0.15
    position_obstacle4 = Point()
    if current_time > 10 and current_time < 40:
        position_obstacle4.x = 5.5-0.15*(current_time-10) 
    elif current_time >=40  :
        position_obstacle4.x=0.8

    else:
        position_obstacle4.x = 5.5

    position_obstacle4.y = 3

    position_obstacle5 = Point()
    if current_time > 30:
        position_obstacle5.x = 4.95-0.04*(current_time-30) 
        position_obstacle5.y = 4.6-0.01*(current_time-30)
    else :
        position_obstacle5.x = 4.95
        position_obstacle5.y = 4.6


    position_obstacle11 = Point()
    if current_time > 40:
        position_obstacle11.x = 7 
        position_obstacle11.y = 7-0.025*(current_time-40)
    else :
        position_obstacle11.x = 7
        position_obstacle11.y = 7

    
    model_state_obstacle1 = ModelState()
    model_state_obstacle1.model_name = orbstacle1_name
        
    model_state_obstacle1.pose.position = position_obstacle1

    model_state_obstacle4 = ModelState()
    model_state_obstacle4.model_name = orbstacle4_name
        
    model_state_obstacle4.pose.position = position_obstacle4

    model_state_obstacle5 = ModelState()
    model_state_obstacle5.model_name = orbstacle5_name
        
    model_state_obstacle5.pose.position = position_obstacle5

    model_state_obstacle11 = ModelState()
    model_state_obstacle11.model_name = orbstacl11_name
        
    model_state_obstacle11.pose.position = position_obstacle11

    set_model_state_obstacle1(model_state_obstacle1)
    set_model_state_obstacle1(model_state_obstacle4)
    set_model_state_obstacle1(model_state_obstacle5)
    set_model_state_obstacle1(model_state_obstacle11)
    

    rate.sleep()

