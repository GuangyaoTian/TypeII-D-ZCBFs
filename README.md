# Type II D-ZCBFs
This program demonstrates a novel algorithm that combines the advantages of Model Predictive Control (MPC) with Type II Discrete Control Barrier Functions (D-CBFs), enabling a vehicle to navigate around obstacles and reach its destination. The simulation environment is based on Ubuntu 20.04, ROS Noetic, and Gazebo 11.For more detailed information, please refer to the accompanying article "Online Efficient Safety-Critical Control for Mobile Robots in Unknown
Dynamic Multi-Obstacle Environments".

## Installation
Creating a workspace:
```
mkdir -p catkin_ws/src
```
```
cd catkin_ws/src
```
```
source /opt/ros/noetic/setup.bash
```
```
catkin_init_workspace
```
To use this project,you need to download some dependency packages about Turtlebot3:
```
sudo apt install ros-noetic-turtlebot3-msgs
```
```
sudo apt install ros-noetic-turtlebot3
```
```
git clone https://github.com/ROBOTIS-GIT/turtlebot3
```
```
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs
```
```
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
```
```
git clone https://github.com/GuangyaoTian/TypeII-D-ZCBFs.git
```
Afterwards, it's necessary to copy the worlds and launch files into the turtlebot3_simulations.
```
cp -r TypeII-D-ZCBFs/worlds/* turtlebot3_simulations/turtlebot3_gazebo/worlds/
```
```
cp -r TypeII-D-ZCBFs/launch/* turtlebot3_simulations/turtlebot3_gazebo/launch/
```
Initializing the workspace:
```
cd ..
```
```
catkin_make
```
Running the code that you want:
```
source devel/setup.bash
```
Select the model name of Turtlebot. Here waffle_pi is used:
```
export TURTLEBOT3_MODEL=waffle_pi
```
Run the gazebo environment that will operate the Turtlebot (for example one obstacle in world):
```
roslaunch turtlebot3_gazebo turtlebot3_one_obs_world.launch
```
run rviz program:
```
roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch
```
run camera:
```
rosrun image_view image_view image:=/camera/rgb/image_raw
```
run the python file that you want:
```
python3 TypeII-D-ZCBFs_one_obs.py
```
## Result

## License
The contents of this repository are covered under the [MIT License](LICENSE).
## Reference
Turtlebot3 gazebo simulation
https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
