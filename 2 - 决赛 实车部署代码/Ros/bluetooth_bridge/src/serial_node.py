#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import sys
import signal
import serial
import time
import threading
from std_msgs.msg import Int32, String
from bluetooth_bridge.msg import Sensors
from serial_utils import decode_data

""" sub topics """
# control msg from bluetooth
topic_from_bluetooth = "/bluetooth/received/data"

# control msg from auto_driver
topic_control_direction = "/auto_driver/send/direction"
topic_control_speed     = "/auto_driver/send/speed"
topic_control_mode      = "/auto_driver/send/mode"
topic_control_beep      = "/auto_driver/send/beep"

""" pub topics """
# sensor msg
topic_sensors  = '/vcu'  # 传感器数据topic

# raw data from vcu, send to bluetooth
topic_vcu_data = "/bluetooth/send"

""" serial """
def sigint_handler(signal, frame):
        print("Interrupt!\nTerminated")
        if ser.isOpen():
            ser.close()
        sys.exit(0)

def serial_init():
    serialPort = "/dev/ttyUSB0"
    baudRate = 1000000
    ser = serial.Serial(serialPort, baudRate)
    print("[Init Serial]: Serial port is %s, baudRate is %d" % (serialPort, baudRate))
    time.sleep(1)
    return ser

ser = serial_init()
signal.signal(signal.SIGINT, sigint_handler)


""" callbacks """
# default       head,direction,speed,mode,manul,beep,nouse,crc
#             [ 0xaa,50       ,0    ,3   ,0    ,0   ,0  , 0]
driver_data = [chr(0xaa), chr(50), chr(0), chr(3), chr(0), chr(0), chr(0), chr(0)]
flag_manul = 0

def calc_crc(data):
    return chr(ord(data[0])^ord(data[1])^ord(data[2])^ord(data[3])^ord(data[4])^ord(data[5])^ord(data[6]))

def callback_bluetooth(message):
    global flag_manul
    flag_manul = ord(message.data[4])     # update flag_manul
    if flag_manul == 1:
        ser.write(message.data[0:8])
        ser.flush()
        ser.write(message.data[0:8])
        ser.flush()

def callback_direction(message):
    driver_data[1] = chr(message.data)    # update global driver_data
    driver_data[7] = calc_crc(driver_data)
    if flag_manul==0:
        msg = ''.join(driver_data)
        ser.write(msg)
        ser.flush()
        ser.write(msg)
        ser.flush()

def callback_speed(message):
    driver_data[2] = chr(message.data)    # update global driver_data
    driver_data[7] = calc_crc(driver_data)
    if flag_manul==0:
        msg = ''.join(driver_data)
        ser.write(msg)
        ser.flush()
        ser.write(msg)
        ser.flush()

def callback_mode(message):
    driver_data[3] = chr(message.data)    # update global driver_data
    driver_data[7] = calc_crc(driver_data)
    if flag_manul==0:
        msg = ''.join(driver_data)
        ser.write(msg)
        ser.flush()
        ser.write(msg)
        ser.flush()

def callback_beep(message):
    driver_data[5] = chr(message.data)    # update global driver_data
    driver_data[7] = calc_crc(driver_data)
    if flag_manul==0:
        msg = ''.join(driver_data)
        ser.write(msg)
        ser.flush()
        ser.write(msg)
        ser.flush()

def main(hz=100):
    global ser
    #--- node init
    rospy.init_node('serial_node', anonymous=True)
    print("[Serial Node]: Init")

    #--- publish topic
    sensors_pub = rospy.Publisher(topic_sensors, Sensors, queue_size=10)
    vcu_data_pub = rospy.Publisher(topic_vcu_data, String, queue_size = 10)

    #--- subscriber topic
    rospy.Subscriber(topic_from_bluetooth, String, callback_bluetooth)
    rospy.Subscriber(topic_control_direction, Int32, callback_direction)
    rospy.Subscriber(topic_control_speed, Int32, callback_speed)
    rospy.Subscriber(topic_control_mode, Int32, callback_mode)
    rospy.Subscriber(topic_control_beep, Int32, callback_beep)

    ros_spin = threading.Thread(target = rospy.spin)
    ros_spin.start()

    rate = rospy.Rate(hz)
    count=0
    while not rospy.is_shutdown():
        try:
            if count!=0:
                ser = serial.Serial("/dev/ttyUSB0", 1000000)
                ser.timeout==2
            count+=1

            # read serial
            raw_data = ser.read(34)
            ser.flushInput()
            sensor_data = decode_data(raw_data)
            vcu_data_pub.publish(raw_data[0:])
            sensors_pub.publish(sensor_data)

        except Exception, e:
            print("have a serial error")
            if ser.isOpen():
                ser.close()

        rate.sleep()

if __name__ == '__main__':
    main(hz=100)


