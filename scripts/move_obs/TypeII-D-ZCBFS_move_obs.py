#!/usr/bin/env python

import casadi as ca
import cvxpy as cp
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import rospy
import time

from geometry_msgs.msg import Point, Twist
from math import atan2,dist,sqrt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.covariance import EllipticEnvelope
from matplotlib.patches import Ellipse
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

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
formatted_list=[]

## get the information of obstacles from the topic /obstacle_info
def obstacle_info_callback(data):
    global formatted_list
    global last_marker_ids
    obstacle_info = data.data

    formatted_list = extract_and_format_data(obstacle_info)


def extract_and_format_data(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    formatted_data = []
    for i in range(0, len(numbers), 5):
        obstacle_data = [round(float(numbers[i]), 2), round(float(numbers[i+1]), 2),
                         round(float(numbers[i+2]), 2), round(float(numbers[i+3]), 2),
                         round(float(numbers[i+4]), 2)]
        formatted_data.append(obstacle_data)
    return formatted_data

rospy.init_node("pi_controller")
sub = rospy.Subscriber("/odom", Odometry, newOdom)
sub2 =rospy.Subscriber("/obstacle_info", String, obstacle_info_callback)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

speed = Twist()
###MPC
T = 0.1 
N = 30

rob_diam = 0.3 
v_max = 0.2 
omega_max = np.pi/8.0 
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

Q = np.array([[15.0, 0.0, 0.0],[0.0, 15.0, 0.0],[0.0, 0.0, .005]])
R = np.array([[2, 0.0], [0.0, 0.05]])

obj = 0 
for i in range(N):
    obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
cost_func = ca.Function('cost_func', [U, P], [obj], ['control_input', 'params'], ['cost'])
cost_values=[]
g = [] 

execution_times=[]

u0_real_list=[]
u1_real_list=[]

nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vertcat(*g)}
opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

lbg = [] 
ubg = [] 

lbx = [] 
ubx = [] 


for _ in range(N):
    lbx.append(0)
    ubx.append(v_max)
    lbx.append(-omega_max)
    ubx.append(omega_max)



t0 = 0.0 
x_idk = np.array([0,0,0]).reshape(-1, 1)
xs = np.array([6,6,0]).reshape(-1, 1) ##target_point

u0 = np.array([0.0, 0.0]*N).reshape(-1, 2) 
    									  
x_c = [] 
u_c = [] 
t_c = [] 
xx = [] 
sim_time = 20.0 
index_t = [] 

mpciter = 0 
start_time = time.time() 
### target point
target_points = [
        np.array([6,6,0]).reshape(-1, 1),
    ]
current_target_index = 0

x_path=[]
y_path=[]
theta_path=[]
t=[]

car_circle=0
start_time3 = time.time()
T_zong_list=[]
min_dist_list=[]

flag = False
Rate = rospy.Rate(10) 


while not flag: 
    if current_target_index<1 :
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
        if len (formatted_list) > 0:
            stop=0
            distlist=[]
            for m in range(len(formatted_list)):
                obs=formatted_list[m]
                obs_x=obs[0]
                obs_y=obs[1]
                obs_r=obs[2]
                obs_vx=obs[3]
                obs_vy=obs[4]
                ###  get information from obstacle list

                distance=((x_real+l*np.cos(theta_real)-obs_x)**2+(y_real-obs_y+l*np.sin(theta_real))**2)**0.5-obs_r-0.25
                distlist.append(distance)
                if distance <=0.4:
                    CBF_Condition=distance
                    Q_cbf=np.array([[1000, 0], [0, 1]])
                    c_cbf=np.zeros(2)

                    if distance<=0.2:

                        obs_rel_y_1=(obs_y-y_real-obs_vy*1)*np.cos(theta_real)-(obs_x-x_real-obs_vx*1)*np.sin(theta_real)
                        obs_rel_y_2=(obs_y+obs_vy*1-y_real)*np.cos(theta_real)-(obs_x-x_real+obs_vx*1)*np.sin(theta_real)
                        obs_rel_y=(obs_y-y_real)*np.cos(theta_real)-(obs_x-x_real)*np.sin(theta_real)
                        obs_mul=obs_rel_y_1*obs_rel_y_2

                        obs_rel_vy=obs_vy*np.cos(theta_real)-obs_vx*np.sin(theta_real)


                        if obs_rel_y > 0 and obs_rel_y < 0.2 and obs_rel_vy < -0.05:
                            stop=1
                        if obs_rel_y < 0 and obs_rel_y > -0.2 and obs_rel_vy >0.05 :
                            stop=1

                        if obs_mul< 0:
                            stop=1

                    
                    e1=-0.5*((x_real+l*np.cos(theta_real)-obs_x)**2+(y_real-obs_y+l*np.sin(theta_real))**2)**(-0.5)*2*(((x_real+l*np.cos(theta_real)-obs_x))*np.cos(theta_real)+(y_real-obs_y+l*np.sin(theta_real))*np.sin(theta_real))
                    e2=-0.5*((x_real+l*np.cos(theta_real)-obs_x)**2+(y_real-obs_y+l*np.sin(theta_real))**2)**(-0.5)*2*(((x_real+l*np.cos(theta_real)-obs_x))*np.sin(theta_real)*l*(-1)+(y_real-obs_y+l*np.sin(theta_real))*np.cos(theta_real)*l)
                    ### CBF for move obstacle
                    e3=0.5*((x_real+l*np.cos(theta_real)-obs_x)**2+(y_real-obs_y+l*np.sin(theta_real))**2)**(-0.5)*2*(((x_real+l*np.cos(theta_real)-obs_x))*obs_vx+(y_real-obs_y+l*np.sin(theta_real))*obs_vy)
                    CBF_Condition=distance-e3
                    A = np.array([[e1, e2],[1,0],[0,1],[0,-1]])
                    b_cbf=np.array([CBF_Condition,v_max,omega_max,omega_max])
                    b_cbf=b_cbf.reshape(-1, 1)
                    u1 = cp.Variable()
                    u2 = cp.Variable()
                    if u_real[0]>=0:
                        objective = cp.Minimize(0.5 * cp.quad_form(cp.vstack([u1-0.2, u2]), Q_cbf) + c_cbf @ cp.vstack([u1, u2]))
                    else:
                        objective = cp.Minimize(0.5 * cp.quad_form(cp.vstack([u1+0.2, u2]), Q_cbf) + c_cbf @ cp.vstack([u1, u2]))
                    constraints = [cp.matmul(A, cp.vstack([u1, u2])) <= b_cbf]
                    problem = cp.Problem(objective, constraints)
                    problem.solve()
                    rate=distance/0.4
                    ### Combine the CBF and MPC
                    if rate>1:
                        u_real[0]=u_real[0]
                        u_real[1]=u_real[1]

                    elif rate<= 0:
                        u_real[0] =u1.value
                        u_real[1] =u2.value
                    else:
                        u_real[0] = u1.value*(1-rate)+rate*u_real[0]
                        u_real[1] = u2.value*(1-rate)+rate*u_real[1] 
                if stop ==1:
                    u_real[0]=0
                    u_real[1]=0
            mindist=min(distlist)
            min_dist_list.append(mindist)
        speed.linear.x = u_real[0]
        speed.angular.z = u_real[1]
        x_path.append(x_real)
        y_path.append(y_real)
        theta_path.append(theta_real)
        print(theta_real)
        u0_real_list.append(u_real.full()[0][0])
        u1_real_list.append(u_real.full()[1][0])
        current_cost = cost_func(u_sol, c_p)
        cost_values.append(float(current_cost))
        pub.publish(speed)
        if np.linalg.norm(x_idk-xs)<0.05:
            current_target_index=current_target_index+1


    else:
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
