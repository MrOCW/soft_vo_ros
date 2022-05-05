cmake_minimum_required(VERSION 3.0.2)
project(soft_vo_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package( OpenCV 4.5 REQUIRED 
  PATHS /usr/local
  NO_DEFAULT_PATH)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  message_filters
  nav_msgs
  roscpp
  std_msgs
  tf
)

add_subdirectory(src/evaluate/)

include_directories(src/evaluate)

if(USE_CUDA)
    find_package(CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
    add_definitions(-DUSE_CUDA)
endif(USE_CUDA)
###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
##  INCLUDE_DIRS include
##  LIBRARIES soft_vo_ros
##  CATKIN_DEPENDS cv_bridge geometry_msgs message_filters nav_msgs roscpp std_msgs tf
##  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_library(feature SHARED src/feature.cpp)
add_library(bucket SHARED src/bucket.cpp)
add_library(utils SHARED src/utils.cpp)
add_library(visualOdometry SHARED src/visualOdometry.cpp)
add_library(rgbd_standalone SHARED src/rgbd_standalone.cpp)
add_library(Frame SHARED src/Frame.cpp)
add_library(VisualOdometer SHARED src/VisualOdometer.cpp)
add_executable( run src/main.cpp )
add_executable( vo_node src/visual_odom_node.cpp )

target_link_libraries( bucket ${OpenCV_LIBS} )
target_link_libraries( feature ${OpenCV_LIBS} bucket)
target_link_libraries( utils evaluate_odometry feature ${OpenCV_LIBS} )
target_link_libraries( visualOdometry utils bucket feature ${OpenCV_LIBS})
target_link_libraries( VisualOdometer visualOdometry ${catkin_LIBRARIES})

target_link_libraries( Frame ${OpenCV_LIBS} )
target_link_libraries( run ${OpenCV_LIBS} bucket feature utils visualOdometry Frame rgbd_standalone )
target_link_libraries( vo_node ${OpenCV_LIBS} bucket feature utils visualOdometry Frame ${catkin_LIBRARIES} VisualOdometer)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
