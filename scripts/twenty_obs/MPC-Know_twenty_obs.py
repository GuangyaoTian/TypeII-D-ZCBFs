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
from tf.transformations import euler_from_quaternion

### get the status of turtlrbot3 from odom
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



rospy.init_node("Turtlebot_controller")
sub = rospy.Subscriber("/odom", Odometry, newOdom)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

speed = Twist()

T = 0.1 
N = 30 

rob_diam = 0.3 
v_max = 0.2 
omega_max = np.pi/8.0 
### Modelling
x = ca.SX.sym('x') 
y = ca.SX.sym('y') 
theta = ca.SX.sym('theta') 
states = ca.vertcat(x, y) 
states = ca.vertcat(states, theta) 
n_states = states.size()[0] 
v = ca.SX.sym('v') 
omega = ca.SX.sym('omega') 
controls = ca.vertcat(v, omega) 
n_controls = controls.size()[0] 
l=0.05
# rhs
rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta))
rhs = ca.vertcat(rhs, omega)
f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

U = ca.SX.sym('U', n_controls, N) 
X = ca.SX.sym('X', n_states, N+1) 
P = ca.SX.sym('P', n_states+n_states) 

X[:, 0] = P[:3] 
for i in range(N):
    f_value = f(X[:, i], U[:, i]) 
    X[:, i+1] = X[:, i] + f_value*T

ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])
## NLP Problem
Q = np.array([[15.0, 0.0, 0.0],[0.0, 15.0, 0.0],[0.0, 0.0, .005]])
R = np.array([[2, 0.0], [0.0, 0.05]])
### cost
obj = 0 
for i in range(N):
    obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
cost_func = ca.Function('cost_func', [U, P], [obj], ['control_input', 'params'], ['cost'])
cost_values=[]
### constraints
g = [] 


x_obs_list=np.array([1.5,0.75,0.75,2.25,2.25,-1,1.5,0.75,-1,2.9,0.1,1.5,4,-1,4,2.25,4,4,2.25,-1,0.75])
y_obs_list=np.array([1.5,0.75,2.25,0.75,2.25,-1,0.05,-1,4,1.5,1.5,3,0.75,0.75,2.25,-1,4,-1,4,2.25,4])
### the information of the obstacles are knowen before the turtlebot work

for i in range(N+1):
    for m in range(21):
        g.append(ca.sqrt((X[0, i]-x_obs_list[m])**2+(X[1, i]-y_obs_list[m])**2)) # should be smaller als 0.0
    
nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
### ipot
opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
## solver
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
### constraints for states
lbg = []
ubg = []
### constraints for input
lbx = [] 
ubx = [] 


for _ in range(N+1):
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)
    lbg.append(0.4)
    ubg.append(np.inf)



for _ in range(N):
    lbx.append(-v_max/2)
    ubx.append(v_max)
    lbx.append(-omega_max)
    ubx.append(omega_max)


t0 = 0.0 
x_idk = np.array([0,0,0]).reshape(-1, 1)
xs = np.array([3,4,0]).reshape(-1, 1) ### first target points
u0 = np.array([0.0, 0.0]*N).reshape(-1, 2) 
x_c = [] 
u_c = [] 
t_c = []
xx = [] 
index_t = [] 

start_time = time.time() 

### target points
target_points = [
        np.array([3,3,-np.pi/2]).reshape(-1, 1),
        np.array([3,0,-np.pi]).reshape(-1, 1),
        np.array([0,3,-np.pi/2]).reshape(-1, 1),
        np.array([0,0,0]).reshape(-1, 1)
    ]
current_target_index = 0
### save data 
x_path=[]
y_path=[]
theta_path=[]
t=[]
flag = False
Rate = rospy.Rate(10) 
execution_times=[]
u0_real_list=[]
u1_real_list=[]
start_time3 = time.time()
T_zong_list=[]
min_dist_list=[]

while not flag: 
    if current_target_index<4:
        xs = target_points[current_target_index]
        start_time = time.time()
        x_idk = np.array([x_real,y_real,theta_real]).reshape(-1, 1)
        c_p = np.concatenate((x_idk, xs)) 
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'], n_controls, N) 
        ff_value = ff(u_sol, c_p) 
        u_real=u_sol[:, 0]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        distlist=[]

        for m in range(10):
            distlist.append(ca.sqrt((x_real-x_obs_list[m])**2+(y_real-y_obs_list[m])**2)) 

        mindist=min(distlist)
        min_dist_list.append(mindist)

        speed.linear.x = u_real[0]
        speed.angular.z = u_real[1]
        x_path.append(x_real)
        y_path.append(y_real)
        theta_path.append(theta_real)

        u0_real_list.append(u_real.full()[0][0])
        u1_real_list.append(u_real.full()[1][0])

        current_cost = cost_func(u_sol, c_p)
        cost_values.append(float(current_cost))
        pub.publish(speed)
        if np.linalg.norm(x_idk-xs)<0.05:
            ### go to next target point
            current_target_index=current_target_index+1


    else:
        ### finished and stopped
        speed.linear.x = 0
        speed.angular.z = 0
        pub.publish(speed)
        flag= True

    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    t_zong=end_time-start_time3
    T_zong_list.append(t_zong)
    Rate.sleep()

average_execution_time = sum(execution_times) / len(execution_times)

###save data in json

xy_data={"x_path": x_path,
         "y_path": y_path,
         "theta_path": theta_path,
         "T": T_zong_list
        }
with open("xy_path_data.json", "w") as file:
    json.dump(xy_data, file)

t_data ={"t_path": execution_times,
          "T": T_zong_list
        }
with open("t_path_data.json", "w") as file:
    json.dump(t_data, file)

cost_data={"cost_values": cost_values,
           "T": T_zong_list
          }
with open("costs_values_data.json", "w") as file:
    json.dump(cost_data, file)


input_data = {
    "v": u0_real_list,
    "w": u1_real_list,
    "T": T_zong_list
}

with open("input.json", "w") as file:
    json.dump(input_data, file)

min_data = {
    "min": min_dist_list,
    "T": T_zong_list
}
with open("mindist.json", "w") as file:
    json.dump(min_data, file)

