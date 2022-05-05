
#include <ros/ros.h>
#include "VisualOdometer.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

int main(int argc, char **argv)
{
  

  ros::init(argc, argv, "visual_odom_node");
  ros::NodeHandle nh;

  
  image_transport::ImageTransport it(nh);
  VisualOdometer visual_odometer(nh,it, std::string(argv[1]));

  image_transport::SubscriberFilter left_image_sub(it, "/kitti/camera_color_left/image_raw", 1);
  image_transport::SubscriberFilter right_image_sub(it, "/kitti/camera_color_right/image_raw", 1);
  // image_transport::SubscriberFilter right_cam_info_sub(it, "/zed/right/camera_info", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> left_cam_info_sub(nh, "/kitti/camera_color_left/camera_info", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> right_cam_info_sub(nh, "/kitti/camera_color_right/camera_info", 1);
  

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10),left_image_sub, right_image_sub, left_cam_info_sub, right_cam_info_sub );
  
  sync.registerCallback(boost::bind(&VisualOdometer::imageGrabCallback, &visual_odometer, _1, _2, _3, _4));

  ros::spin();

  return 0;
}