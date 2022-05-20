#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
cd $basepath

rm -r ~/speedstar_ws
mkdir -p ~/speedstar_ws/src
cp -r ./auto_driver/ ./bluetooth_bridge/ ./rplidar_ros/ ~/speedstar_ws/src/
chmod +x ~/speedstar_ws/src/auto_driver/src/*.py ~/speedstar_ws/src/bluetooth_bridge/src/*.py ~/speedstar_ws/src/rplidar_ros/src/*.py

cd ~/speedstar_ws
catkin_make