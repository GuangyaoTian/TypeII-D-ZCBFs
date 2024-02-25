# Type II D-ZCBFs
This program demonstrates a novel algorithm that combines the advantages of Model Predictive Control (MPC) with Type II Discrete Control Barrier Functions (D-CBFs), enabling a vehicle to navigate around obstacles and reach its destination. The simulation environment is based on Ubuntu 20.04, ROS Noetic, and Gazebo 11.For more detailed information, please refer to the accompanying article "Online Efficient Safety-Critical Control for Mobile Robots in Unknown
Dynamic Multi-Obstacle Environments".

## Installation
To use this project,you need to download some dependency packages about Turtlebot3
```
mkdir -p catkin_ws/src
```
```
cd catkin_ws/src
```
> source /opt/ros/noetic/setup.bash
> 
> catkin_init_workspace
> sudo apt install ros-noetic-turtlebot3-msgs
>
> sudo apt install ros-noetic-turtlebot3
>
> git clone https://github.com/ROBOTIS-GIT/turtlebot3
>
> git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs
>
> git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
>
> git clone https://github.com/GuangyaoTian/TypeII-D-ZCBFs.git
## Reference
Turtlebot3 gazebo simulation
https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
