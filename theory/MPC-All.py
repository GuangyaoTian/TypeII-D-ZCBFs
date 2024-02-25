#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time
from draw import Draw_Obstacle

def shift_movement(T, t0, x0, u, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = ca.horzcat(u[:, 1:], u[:, -1])

    return t, st, u_end.T

if __name__ == '__main__':
    T = 0.2 # sampling time [s]
    N = 100 # prediction horizon
    rob_diam = 0.3 
    v_max = 0.6
    omega_max = np.pi/4.0

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

    ## rhs
    rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta))
    rhs = ca.vertcat(rhs, omega)

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N+1)
    P = ca.SX.sym('P', n_states+n_states)


    ### define
    X[:, 0] = P[:3] # initial condiction

    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[:, i], U[:, i])
        X[:, i+1] = X[:, i] + f_value*T

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        obj = obj + (X[:, i]-P[3:]).T @ Q @ (X[:, i]-P[3:]) + U[:, i].T @ R @ U[:, i]

    #### constrains
    g = [] 

    
    x_obs_list=np.array([0.1,1,2.2,0.8])
    y_obs_list=np.array([1,2.1,1,0.1])


    for i in range(N+1):
        for n in range(4):
            g.append(ca.sqrt((X[0, i]-x_obs_list[n])**2+(X[1, i]-y_obs_list[n])**2))
        

    nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vcat(g)} 
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
    
    execution_times = []
    
    # Simulation
    lbg = []
    ubg = []
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

    for _ in range(N):
        lbx.append(-v_max)
        ubx.append(v_max)
        lbx.append(-omega_max)
        ubx.append(omega_max)
    t0 = 0.0
    x0 = np.array([0.0, 0.0, np.pi/2]).reshape(-1, 1)# initial state
    xs = np.array([0, 2, 0.0]).reshape(-1, 1) # final state
    u0 = np.array([0.0, 0.0]*N).reshape(-1, 2)# np.ones((N, 2)) # controls
    x_c = [] 
    u_c = []
    t_c = [] 
    xx = []
    sim_time = 40.0

    target_states = [
        np.array([0, 2, 0]).reshape(-1, 1),
        np.array([2, 2, -np.pi/2]).reshape(-1, 1),
        np.array([2, 0, -np.pi]).reshape(-1, 1),
        np.array([0, 0, -3*np.pi/2]).reshape(-1, 1)
    ]

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    c_p = np.concatenate((x0, xs))
    init_control = ca.reshape(u0, -1, 1)
    res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    lam_x_ = res['lam_x']
    ### inital test
    while(len(target_states) > 0):

        xs = target_states.pop(0)
        while np.linalg.norm(x0-xs) > 1e-1 and mpciter < sim_time/T:
            start_time = time.time()
            ## set parameter
            c_p = np.concatenate((x0, xs))
            init_control = ca.reshape(u0, -1, 1)
            t_ = time.time()
            res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx, lam_x0=lam_x_)
            lam_x_ = res['lam_x']
            index_t.append(time.time()- t_)
            u_sol = ca.reshape(res['x'], n_controls, N) 
            ff_value = ff(u_sol, c_p) 
            x_c.append(ff_value)
            u_c.append(u_sol[:, 0])
            t_c.append(t0)

            t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)
            x0 = ca.reshape(x0, -1, 1)
            xx.append(x0.full())
            mpciter = mpciter + 1
            end_time = time.time()
            execution_time_iteration = end_time - start_time
            execution_times.append(execution_time_iteration)
    t_v = np.array(index_t)

    average_execution_time = sum(execution_times) / len(execution_times)
    draw_result = Draw_Obstacle(rob_diam=0.3, init_state=x0.full(), target_state=xs, robot_states=xx,export_fig=True)