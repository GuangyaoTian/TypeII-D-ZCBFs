#!/usr/bin/env python
# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib as mpl

class Draw_Obstacle(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, 
                 rob_diam=0.3, export_fig=True):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-0.8, 3), ylim=(-0.8, 3.))
        
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        self.ani.save('obstacle.gif', writer='Pillow', fps=10)
        plt.show()

    def animation_init(self):
        # plot target state
        ##self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        ##self.ax.add_artist(self.target_circle)
        ##self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         ##self.rob_radius * np.cos(self.target_state[2]),
                                         ##self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        ##self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='black', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.obstacle_circle = plt.Circle(np.array([1,2]), 0.2, color='blue', fill=True)
        self.ax.add_artist(self.obstacle_circle)
        self.obstacle_circle2 = plt.Circle(np.array([0.1,1]), 0.2, color='b', fill=True)
        self.ax.add_artist(self.obstacle_circle2)
        self.obstacle_circle3 = plt.Circle(np.array([2.2,1]), 0.2, color='b', fill=True)
        self.ax.add_artist(self.obstacle_circle3)
        self.obstacle_circle4 = plt.Circle(np.array([0.8,0.1]), 0.2, color='b', fill=True)
        self.ax.add_artist(self.obstacle_circle4)
        return self.robot_body, self.robot_arr, self.obstacle_circle,self.obstacle_circle2,self.obstacle_circle3,self.obstacle_circle4
        ###return self.target_circle, self.target_arr, self.robot_body, self.robot_arr, self.obstacle_circle,self.obstacle_circle2,self.obstacle_circle3,self.obstacle_circle4
    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body
