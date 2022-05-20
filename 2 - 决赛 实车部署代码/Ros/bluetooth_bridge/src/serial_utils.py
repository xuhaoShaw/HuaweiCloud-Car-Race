#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct
from bluetooth_bridge.msg import Sensors


def decode_data(data):
    """
    将串口读到的原始字节数据解码为Sensors类
    """
    sfmt = struct.Struct('<bh2b13h3b')
    unpack_data = sfmt.unpack(data)
    sensor_data = Sensors(
        MotorSpeed  = unpack_data[1],
        Mode        = unpack_data[2],
        Direction   = unpack_data[3],
        Supersonic  = unpack_data[4],
        ax          = unpack_data[5],
        ay          = unpack_data[6],
        az          = unpack_data[7],
        alphax      = unpack_data[8],
        alphay      = unpack_data[9],
        alphaz      = unpack_data[10],
        Bx          = unpack_data[11],
        By          = unpack_data[12],
        Bz          = unpack_data[13],
        thetax      = unpack_data[14],
        thetay      = unpack_data[15],
        thetaz      = unpack_data[16],
        BatteryVoltage      = unpack_data[17],
        MotorTemperature    = unpack_data[18]
    )
    return sensor_data