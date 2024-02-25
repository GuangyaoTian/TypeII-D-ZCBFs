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

num_ellipses = 0
ellipse_arrays = []
num_ellipses2=0
ellipse_arrays2 = []
l_lida=0.06
###l_lida is the distance between the move center and the center of Lidar (Turtlebot3_Waffle)
def scan_callback(msg):
    global num_ellipses
    global num_ellipses2
    global ellipse_arrays
    global ellipse_arrays2
    angle_min = msg.angle_min
    angle_increment = msg.angle_increment
    points = [] ### get the points of the obstacles from Lidar
    for i, distance in enumerate(msg.ranges):
        if math.isinf(distance) or math.isnan(distance):
            continue
        angle = angle_min + i * angle_increment + theta_real
        x = distance * math.cos(angle)+x_real-l_lida*math.cos(theta_real)
        y = distance * math.sin(angle)+y_real-l_lida*math.sin(theta_real)
        points.append((x, y))

    points_array = np.array(points)
    if points_array.size == 0:
        num_ellipses = 0  
        ellipse_arrays = []  
        return 
    ###DBSCAN
    dbscan = DBSCAN(eps=0.1, min_samples=3)
    clusters = dbscan.fit_predict(points_array)
    X_filtered = points_array[clusters != -1]
    clusters_filtered = clusters[clusters != -1]
    unique_labels = np.unique(clusters_filtered)
    num_ellipses = 0
    ellipse_arrays = []
    ### get the obstacles
    for label in unique_labels:
        points = X_filtered[clusters_filtered == label]
        if len(points) > 0:
            distances = squareform(pdist(points, 'euclidean'))
            farthest_points_idx = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
            point1 = points[farthest_points_idx[0]]
            point2 = points[farthest_points_idx[1]]
            center = (point1 + point2) / 2
            a = np.linalg.norm(point1 - point2) / 2
            b = max(perpendicular_distance(point, point1, point2) for point in points)

            long_axis_vector = point2 - point1
            angle_radians = np.arctan2(long_axis_vector[1], long_axis_vector[0])
            ellipse_array = [center[0], center[1], a , b  , angle_radians] 
            ellipse_arrays.append(ellipse_array)
            num_ellipses += 1  
    ellipse_arrays_np = np.array(ellipse_arrays)
    
    num_ellipses2=num_ellipses
    ellipse_arrays2=ellipse_arrays_np

def perpendicular_distance(point, line_point1, line_point2):
    line_vec = line_point2 - line_point1
    point_vec = point - line_point1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)    
    nearest = line_unitvec * t
    dist = np.linalg.norm(point_vec_scaled - nearest)
    return dist * line_len     

rospy.init_node("Turtlebot_controller")
sub = rospy.Subscriber("/odom", Odometry, newOdom)
sub2 = rospy.Subscriber("/scan", LaserScan, scan_callback)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

speed = Twist()
### MPC
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
l=0.05 ### only use in CBF
### rhs
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
### NLP Problem
Q = np.array([[15.0, 0.0, 0.0],[0.0, 15.0, 0.0],[0.0, 0.0, .005]])
R = np.array([[2, 0.0], [0.0, 0.05]])
### cost
obj = 0 
for i in range(N):
    obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
cost_func = ca.Function('cost_func', [U, P], [obj], ['control_input', 'params'], ['cost'])
cost_values=[]
### Constraints
g = [] 

nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
### ipot
opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
### solver
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
### constraints for states
lbg = [] 
ubg = [] 
### constraints for input
lbx = [] 
ubx = [] 

for _ in range(N):
    lbx.append(-v_max)
    ubx.append(v_max)
    lbx.append(-omega_max)
    ubx.append(omega_max)

t0 = 0.0 
x_idk = np.array([0,0,0]).reshape(-1, 1)
xs = np.array([3,4,0]).reshape(-1, 1) ### first target point
u0 = np.array([0.0, 0.0]*N).reshape(-1, 2) 
x_c = [] 
u_c = [] 
t_c = [] 
xx = [] 
index_t = []
start_time = time.time() 
### target points (in this code,four target points)
target_points = [
        np.array([3,3,-np.pi/2]).reshape(-1, 1),
        np.array([3,0,-np.pi]).reshape(-1, 1),
        np.array([0,3,-np.pi/2]).reshape(-1, 1),
        np.array([0,0,0]).reshape(-1, 1)
    ]
current_target_index = 0

x_path=[]
y_path=[]
theta_path=[]
execution_times=[]
u0_real_list=[]
u1_real_list=[]
t=[]
flag = False
Rate = rospy.Rate(10) 

car_circle=0
start_time3 = time.time()
T_zong_list=[]
min_dist_list=[]
while not flag: 
    if current_target_index<4 :
        xs = target_points[current_target_index]
        start_time = time.time()
        x_idk = np.array([x_real,y_real,theta_real]).reshape(-1, 1)
        c_p = np.concatenate((x_idk, xs)) 
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'], n_controls, N) ### get the input from the MPC only (cannot avoid the obstacle)
        ff_value = ff(u_sol, c_p) 
        u_real=u_sol[:, 0]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        ellipse_arrays_np = np.array(ellipse_arrays2)
        if num_ellipses2 > 0:
            distlist=[]
            for m in range(len(ellipse_arrays_np)):
                obs=ellipse_arrays_np[m]
                obs_x=obs[0]
                obs_y=obs[1]
                distance= ((x_real+l*np.cos(theta_real)-obs_x)**2+(y_real-obs_y+l*np.sin(theta_real))**2)**0.5-obs[2]-0.3
                ###CBF for Ellipse
                x_diff=x_real+l*np.cos(theta_real)-obs_x
                y_diff=y_real+l*np.sin(theta_real)-obs_y
                distan=((x_real-obs_x)**2+(y_real-obs_y)**2)**0.5-obs[2]
                distlist.append(distan)
                obs_a=obs[2]+0.3
                obs_b=obs[3]+0.3
                obs_ang=obs[4]
                x2_diff=x_diff * np.cos(obs_ang) + y_diff * np.sin(obs_ang)
                y2_diff=-x_diff * np.sin(obs_ang) + y_diff * np.cos(obs_ang)
                bili=obs_a/obs_b
                b_shiji=np.sqrt((x_real-obs_x)**2/bili**2+(y_real-obs_y)**2)
                distance2=b_shiji-obs_b
                
                CBF=distance2

                if CBF<=0.3:
                    CBF_Condition=CBF
                    Q_cbf=np.array([[1000, 0], [0, 1]])
                    c_cbf=np.zeros(2)
                    b_cbf=np.array([CBF_Condition,v_max,omega_max,omega_max])
                    b_cbf=b_cbf.reshape(-1, 1)
            
                    e1=-2*x2_diff/obs_a**2*(np.cos(theta_real)*np.cos(obs_ang)+np.sin(theta_real)*np.sin(obs_ang))-2*y2_diff/obs_b**2*(np.sin(theta_real)*np.cos(obs_ang)-np.cos(theta_real)*np.sin(obs_ang))
                    e2=-2*x2_diff/obs_a**2*(np.cos(obs_ang)*(-1)*l*np.sin(theta_real)+np.sin(obs_ang)*l*np.cos(theta_real))-2*y2_diff/obs_b**2*(l*np.sin(theta_real)*np.sin(obs_ang)+np.cos(obs_ang)*l*np.cos(theta_real))
                    A = np.array([[e1, e2],[1,0],[0,1],[0,-1]])
                    u1 = cp.Variable()
                    u2 = cp.Variable()
                    objective = cp.Minimize(0.5 * cp.quad_form(cp.vstack([u1-0.2, u2]), Q_cbf) + c_cbf @ cp.vstack([u1, u2]))
                    constraints = [cp.matmul(A, cp.vstack([u1, u2])) <= b_cbf]
                    problem = cp.Problem(objective, constraints)
                    problem.solve()
                    ###Combine CBF and MPC
                    rate = CBF/0.3
                    if rate>1:
                        u_real[0]=u_real[0]
                        u_real[1]=u_real[1]

                    elif rate<= 0:
                        u_real[0] =u1.value
                        u_real[1] =u2.value
                    else:
                        u_real[0] = u1.value*(1-rate)+rate*u_real[0]
                        u_real[1] = u2.value*(1-rate)+rate*u_real[1] 

            mindist=min(distlist)
            min_dist_list.append(mindist)
        ### input for turtlebot3_waffle
        speed.linear.x = u_real[0]
        speed.angular.z = u_real[1]
        ### save data
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

### save data in json

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
