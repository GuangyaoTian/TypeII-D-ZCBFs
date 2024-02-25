#!/usr/bin/env python

import casadi as ca
import cvxpy as cp
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import rospy
import time

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from math import atan2,sqrt,dist
from matplotlib.patches import Circle,Ellipse
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.covariance import EllipticEnvelope
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

### get the status of turtlrbot3 from odom
obstacle_tracks = []
timestamps = []  
obstacle_info_pub = None

x_real=0
y_real=0
theta_real=0

def newOdom(msg):
    global x_real
    global y_real
    global theta_real

    x_real = msg.pose.pose.position.x
    y_real = msg.pose.pose.position.y

    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta_real) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

def calculate_circle(points):
    if len(points) <= 1:
        return points[0], 0
    hull = ConvexHull(points)
    center = np.mean(points[hull.vertices], axis=0)
    radius = np.max(np.linalg.norm(points[hull.vertices] - center, axis=1))
    return center, radius

def initialize_kalman():
    ### kalman 
    x = np.array([0, 0])  
    P = np.eye(2)  
    F = np.eye(2)  
    H = np.array([[1, 0]])  
    R = np.array([[1]])  
    Q = np.array([[1e-4, 0], [0, 1e-4]])  
    return x, P, F, H, R, Q

def kalman_filter_last_velocity(positions, obstime):
    x, P, F, H, R, Q = initialize_kalman()
    for i in range(1, len(positions)):
        dt = obstime[i] - obstime[i - 1]
        F[0, 1] = dt
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        z = np.array([positions[i]])
        y = z - H.dot(x)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(2) - K.dot(H)).dot(P)
    
    return x[1]  

def calculate_velocity(track, obstime):
    if len(obstime) <= 1:
        return 0, 0  
    else:
        x_positions = track[:, 0]
        x_positions = x_positions-x_positions[0]
        y_positions = track[:, 1]
        y_positions = y_positions-y_positions[0]
        vx= kalman_filter_last_velocity(x_positions, obstime)
        vy= kalman_filter_last_velocity(y_positions, obstime)

        return vx, vy
l_lida=0.06
###l_lida is the distance between the move center and the center of Lidar (Turtlebot3_Waffle)
def scan_callback(scan_msg):
    global obstacle_tracks, timestamps
    current_time = rospy.get_time()
    angle_min = scan_msg.angle_min
    angle_increment = scan_msg.angle_increment

    points = []
    for i, distance in enumerate(scan_msg.ranges):
        if math.isinf(distance) or math.isnan(distance):
            continue
        angle = angle_min + i * angle_increment + theta_real
        x = distance * math.cos(angle)+x_real-l_lida*math.cos(theta_real)
        y = distance * math.sin(angle)+y_real-l_lida*math.sin(theta_real)

        points.append((x, y))
    points=np.array(points)

    ###DBSCAN
    db = DBSCAN(eps=0.1, min_samples=3).fit(points)
    labels = db.labels_
    current_scan = []

    for label in set(labels):
        if label == -1:
            continue  
        cluster_points = points[labels == label]
        center, radius = calculate_circle(cluster_points)
        current_scan.append([center[0], center[1], radius])

    if not obstacle_tracks:  
        obstacle_tracks = [[obstacle] for obstacle in current_scan]
        timestamps = [[current_time] for _ in current_scan] 
    else:
        new_tracks = []
        new_timestamps = []
        matched_current_scan = set()

        for track, time_list in zip(obstacle_tracks, timestamps):
            last_obstacle = track[-1]
            match_found = False

            for j, current_obstacle in enumerate(current_scan):
                if np.linalg.norm(np.array(last_obstacle[:2]) - np.array(current_obstacle[:2])) < 0.2:
                    ### to know the same obstacleor not
                    match_found = True
                    matched_current_scan.add(j)
                    track.append(current_obstacle)
                    time_list.append(current_time)

                    if len(track) > 20: 
                        track.pop(0)
                        time_list.pop(0)
                        ### the list of the position of the same obstacle to calculate the speed
                    break
            
            if match_found:
                new_tracks.append(track)
                new_timestamps.append(time_list)

        for j, obstacle in enumerate(current_scan):
            if j not in matched_current_scan:
                new_tracks.append([obstacle])
                new_timestamps.append([current_time])

        obstacle_tracks = new_tracks
        timestamps = new_timestamps
    
    obstacle_info_list=[]


    for track, obstime in zip(obstacle_tracks, timestamps):
        track_np = np.array(track)
        obstime_np = np.array(obstime)
        vx, vy = calculate_velocity(track_np, obstime_np)
        if vx < 0.01 and vx >-0.01:
            vx=0
        if vy < 0.01 and vy >-0.01:
            vy=0
        ## speed is small the obstacle statics
        obstacle_info = f"position: {track[-1][:2]}, radius: {track[-1][2]}, speed: (x: {vx:.2f}, y: {vy:.2f})"
        obstacle_info_list.append(obstacle_info)
    obstacle_info_str = "\n".join(obstacle_info_list)
    obstacle_info_msg = String(data=obstacle_info_str)
    obstacle_info_pub.publish(obstacle_info_msg)

def listener():
    rospy.init_node('scan_processor', anonymous=True)
    global obstacle_info_pub
    obstacle_info_pub = rospy.Publisher('/obstacle_info', String, queue_size=10)
    rospy.Subscriber("/odom", Odometry, newOdom)
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

