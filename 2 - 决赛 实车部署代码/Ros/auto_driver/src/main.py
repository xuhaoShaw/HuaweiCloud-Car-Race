#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
import sys
from driver_utils import Driver
from driver_event import *
import signal

def sigint_handler(signal, frame):
    print('Terminated!')
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)


def main():
    #--- 小车驱动
    driver = Driver(debug=False)
    #--- 定义事件列表
    follow_lane_event = FollowLaneEvent(driver, timedelay=1)
    red_stop_event = RedStopEvent(driver, scale_prop=0.02, y_limit=0.7, score_limit=0.4)
    green_go_event = GreenGoEvent(driver, scale_prop=0.02, y_limit=0.7, speed=60, score_limit=0.3, go_time=2)
    pedestrian_event = PedestrianEvent(driver, scale_prop=0.1, y_limit=0.5, score_limit=0.8,speed_normal=60,
                                       detect_time=10)
    speed_limited_event = SpeedLimitedEvent(driver, scale_prop=0.01, y_limit=0.8, speed_low=20, speed_normal=60,
                                            score_limit=0.85, max_limited_time=5)
    speed_minimum_event = SpeedMinimumEvent(driver, scale_prop=0.01, y_limit=0.8, speed_normal=60, speed_high=80,
                                            score_limit=0.85)
    obstacle_event = ObstacleEvent(driver, speed_normal=60)
    cross_bridge_event = CrossBridgeEvent(driver, imu_limit=300, speed_limit=80, speed_normal=60, speed_upper=80)
    follow_lidar_event = FollowLidarEvent(driver)
    yellow_back_event = YellowBackEvent(driver, scale_prop=0.02, y_limit=0.7, speed=20, score_limit=0.4, range_limit=450,
                                        turn_time=2, back_direction=35)
    start_end_event = StartEndEvent(driver, scale_prop=0.2, y_limit=0.5, speed=10, score_limit=0.5, range_limit=250,
                                    turn_time=2, back_direction=65)

    event_list = [obstacle_event, red_stop_event, yellow_back_event, pedestrian_event, cross_bridge_event,
                  follow_lidar_event, green_go_event, follow_lane_event, speed_limited_event, speed_minimum_event]  # 默认为优先级排序，越靠前优先级越高

    #--- 主循环
    rate = rospy.Rate(100)  # 循环频率
    event_running = []   # 集合保存正在运行的事件

    while not rospy.is_shutdown():
        # 查询从未开始变化为开始的事件并加入到运行事件列表中
        for i, event in enumerate(event_list):
            if event.is_start() and not i in event_running:
                event_running.append(i)

        # 冒泡算法根据优先级决定运行策略顺序
        for i in range(1, len(event_running)):
            for j in range(0, len(event_running) - i):
                if event_running[j] > event_running[j+1]:
                    event_running[j], event_running[j+1] = event_running[j+1], event_running[j]

        # 遍历执行正在运行事件的策略，并将结束事件删除
        for i in event_running:
            if event_list[i].is_end():
                event_running.remove(i)
            else:
                if i in [0, 1, 2, 3, 4, 5]:
                    event_list[i].strategy()
                    break
                event_list[i].strategy()

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('driver', anonymous=True)
    main()