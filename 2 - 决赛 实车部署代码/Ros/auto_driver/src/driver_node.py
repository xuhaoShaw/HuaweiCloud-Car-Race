#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import signal
import rospy
import numpy as np
import math
from bluetooth_bridge.msg import Sensors
from driver_utils import Driver
import time

def sigint_handler(signal, frame):
    print('Terminated!')
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)


if __name__ == '__main__':

    rospy.init_node('driver', anonymous=True)
    driver = Driver()

    driver.set_mode('D')
    driver.set_direction(10)
    time.sleep(2)
    driver.set_direction(100)
    time.sleep(2)
    driver.set_direction(50)
    time.sleep(2)
    driver.set_speed(100)
    time.sleep(3)
    driver.set_speed(0)
    driver.set_beep(1)
    time.sleep(3)
    driver.set_mode('P')

