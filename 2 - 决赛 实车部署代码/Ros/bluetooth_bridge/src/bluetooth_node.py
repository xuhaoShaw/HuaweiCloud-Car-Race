#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import signal
import bluetooth
import select
import time
from std_msgs.msg import String, Int32
from bluetooth_bridge.msg import BluetoothCtrMsg
import struct
import threading
#--------------------------------- Constants ----------------------------------#

TAG       = "[Bluetooth Bridge Node]: "              ## Node verbosity tag
node_name = "bluetooth_bridge"                    ## ROS node name
version_num="2021/9/14/11/26"

#------------------------------------------------------------------------------#


class Application:

    is_running   = True
    is_connected = False
    bt_channel = 22         # Bluetooth channel
    rate_hz = 100           # 频率
    # subscribe topic
    topic_bt_send = "/bluetooth/send"           # 接收需要发送给蓝牙的信息

    # publish topic
    topic_bt_data    = "/bluetooth/received/data"    # 蓝牙控制指令
    topic_bt_decode  = "/bluetooth/received/decode"  # 蓝牙控制指令解码,便于观察
    topic_status     = "/bluetooth/status"           # 蓝牙状态节点，发布蓝牙状态启动、终止、错误等信息
    topic_sound      = "/soundRequest"               # 声音请求节点


    def __init__(self):
        # Assigning the SIGINT handler
        signal.signal(signal.SIGINT, self.sigint_handler)

        # Starting the node
        rospy.init_node(node_name, anonymous=False)

        # Subscribers
        self.sub_bt_send    = rospy.Subscriber(self.topic_bt_send, String, self.send_callback)

        # Publishers
        self.pub_bt_data    = rospy.Publisher(self.topic_bt_data, String, queue_size=10)
        self.pub_bt_decode  = rospy.Publisher(self.topic_bt_decode, BluetoothCtrMsg, queue_size=10)
        self.pub_sound      = rospy.Publisher(self.topic_sound, Int32, queue_size=10)
        self.pub_status     = rospy.Publisher(self.topic_status, String, queue_size=10)

        time.sleep(0.5)
        self.rate = rospy.Rate(self.rate_hz)
        self.pub_status.publish("INIT")
        self.pub_sound.publish(2)
        self.ros_spin = threading.Thread(target = rospy.spin)
        self.ros_spin.start()

        while self.is_running: #--- connect to bluetooth
            try:
                # Starting the bluetooth server
                self.server_sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
                self.server_sock.bind( ("", self.bt_channel) )
                self.pub_status.publish("LISTENING")
                self.server_sock.listen(1)
                # Accepting incoming connection
                self.client_sock, self.address = self.server_sock.accept()
                self.pub_status.publish("CONNECTED: " + str(self.address))

                self.is_connected = True
                while self.is_running: #--- receive data
                    ready = select.select([self.client_sock],[],[], 2)
                    if ready[0]:
                        data = self.client_sock.recv(1024)

                        if self.check_crc(data): # pub control msg
                            self.pub_bt_data.publish(data)
                            self.pub_bt_decode.publish(self.decode_bt_data(data))
                            if (ord(data[5])):
                                self.pub_sound.publish(1)
                        else:
                            print "CRC not pass"

            except Exception, e:
                self.is_connected = False
                self.server_sock.close()
                self.pub_status.publish("EXCEPTION: " + str(e))
                self.pub_bt_decode.publish(BluetoothCtrMsg())

            self.rate.sleep()

    def sigint_handler(self, signal, frame):
        """ SIGINT Signal handler """
        print TAG,"Interrupt!"
        self.pub_status.publish("SIGINT")
        self.is_running = False
        sys.exit(0)

    def send_callback(self, message):
        """ 将串口读取数据发送给蓝牙端 """
        if self.is_connected:
            self.client_sock.send(message.data)

    def check_crc(self, data):
        """ 检查crc """
        return (ord(data[0])==0xaa) and \
            (ord(data[7])==(ord(data[0])^ord(data[1])^ord(data[2])^ord(data[3])^ord(data[4])^ord(data[5])^ord(data[6])))

    def decode_bt_data(self, data):
        """ 解码蓝牙控制指令 """
        return BluetoothCtrMsg( direction   = ord(data[1]),
                                speed       = ord(data[2]),
                                mode        = ord(data[3]),
                                manual      = ord(data[4]),
                                beep        = ord(data[5]))


#------------------------------------- Main -------------------------------------#

if __name__ == '__main__':
    print TAG,"Started"
    app = Application()

