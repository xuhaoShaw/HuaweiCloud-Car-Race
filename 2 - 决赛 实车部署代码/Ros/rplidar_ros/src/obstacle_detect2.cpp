#include “ros/ros.h”
#include “sensor_msgs/LaserScan.h”
#include <stdlib.h>
#include <stdio.h>
#define RAD2DEG(x) ((x)*180./M_PI)

int stopFlag=0;

void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
{
    int count = scan->scan_time / scan->time_increment;

    ROS_INFO("I heard a laser scan %s[%d]:", scan->header.frame_id.c_str(), count);
    ROS_INFO("angle_range, %f, %f", RAD2DEG(scan->angle_min), RAD2DEG(scan->angle_max));

    for(int i = 0; i < count; i++) {
        float degree = RAD2DEG(scan->angle_min + scan->angle_increment * i);
        if(degree >=-30 && degree <=30 ){
            ROS_INFO(": [%f, %f]", degree, scan->ranges[i]);
            if (scan->ranges[i] <= 0.8) {
                stopFlag = 1;
            }
            else stopFlag = 0;
        }
        else {
            continue;
        }
    }

int main(int argc, char **argv)
{
    ros::init(argc, argv, “rplidar_node_client”);
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe<sensor_msgs::LaserScan>("/scan", 1000, scanCallback);

    ros::Publisher pub=n.advertise<geometry_msgs::Twist>("/is_obstacle",1000);
    ros::Rate loop_rate(15);

    while(ros::ok())
    {
        ros::spinOnce();
        bool is_obs;
        if(stopFlag == 1) { // stop
            is_obs = true;
        }

        else if (stopFlag == 0) {  //go
            is_obs = false;
        }
        pub.publish(is_obs);
        ROS_INFO("是否障碍物？1是，2否",stopFlag);
        loop_rate.sleep();
    } 
    return 0;
}

