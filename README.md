## Stereo Visual Odometry
![example workflow](https://github.com/ZhenghaoFei/visual_odom/actions/workflows/cmake.yml/badge.svg)

This repository is C++ OpenCV implementation of Stereo Visual Odometry, using OpenCV `calcOpticalFlowPyrLK` for feature tracking.

Reference Paper: https://lamor.fer.hr/images/50020776/Cvisic2017.pdf

Demo video: https://www.youtube.com/watch?v=Z3S5J_BHQVw&t=17s

![alt text](images/features.png "features")

![alt text](images/trajectory.png "trajectory")

### Requirements
[OpenCV 4](https://opencv.org/)  
If you are not using CUDA:  
```bash
sudo apt update
sudo apt install libopencv-dev 
```
If you use CUDA, compile and install CUDA enabled OpenCV.

### Dataset
Tested on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) odometry dataset.

### Compile & Run
```bash
cd <your_ws>/src
git clone https://github.com/MrOCW/soft_vo_ros
cd ..
catkin_make
```

### Run VO Node
```bash
rosrun soft_vo_ros vo_node
```  
Image with feature tracking is published on /feature_image topic  
Currently, ROS version does not perform as well as standalone version for some reason  


### GPU CUDA acceleration
To enable GPU acceleration
1. Make sure you have CUDA compatible GPU.
2. Install CUDA, compile and install CUDA supported OpenCV 
3. When compiling, use 
```bash
catkin_make -DUSE_CUDA=ON
```
4. Compile & Run

### Reference code
1. [Monocular visual odometry algorithm](https://github.com/avisingh599/mono-vo/blob/master/README.md)

2. [Matlab implementation of SOFT](https://github.com/Mayankm96/Stereo-Odometry-SOFT/blob/master/README.md)

3. [C++ implementation of SOFT](https://github.com/ZhenghaoFei/visual_odom)  