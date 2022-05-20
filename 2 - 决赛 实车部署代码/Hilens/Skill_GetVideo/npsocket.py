# -*- coding: utf-8 -*-

import socket
import numpy as np
import pickle

class NumpySocket():
    def __init__(self):
        self.address = ''
        self.port = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.type = None  # server or client

    def initialize_sender(self, address, port):
        """
        :param address: host address of the socket e.g 'localhost' or your ip
        :type address: str
        :param port: port in which the socket should be intialized. e.g 4000
        :type port: int
        :return: None
        :rtype: None
        """
        self.address = address
        self. port = port
        self.socket.connect((self.address, self.port))

    def send_array(self, np_array):
        """
        :param np_array: Numpy array to send to the listening socket
        :type np_array: ndarray
        :return: None
        :rtype: None
        """
        data = pickle.dumps(np_array)

        # Send message length first
        message_size = str(len(data)).ljust(16).encode()
        # Then data
        self.socket.sendall(message_size + data)


    def initalize_receiver(self, address, port):
        """
        :param port: port to listen
        :type port: int
        :return: numpy array
        :rtype: ndarray
        """
        self.address = address
        self.port = port
        self.socket.bind((self.address, self.port))
        print('Socket bind complete')
        self.socket.listen()
        self.payload_size = 16  ### CHANGED
        self.data = b''
        self.conn_state = False  # 连接状态


    def receive_array(self):
        """
        等待，直到读到一个完整数组或者超时,
        读到数组则返回: True, 数组
        超时则返回: False, 空数组
        """
        if self.conn_state == False:
            print('Listening...')
            self.conn, _ = self.socket.accept()
            self.conn_state = True

        wait_times = 0
        while len(self.data) < self.payload_size:
            self.data += self.conn.recv(4096)
            wait_times += 1
            if(wait_times > 100): # 连接断开，重新开始accept
                print("Socket break...")
                self.data =  b''
                self.conn_state = False
                return self.conn_state, np.array(0)

        msg_size = int(self.data[:self.payload_size].decode())
        self.data = self.data[self.payload_size:]

        # Retrieve all data based on message size
        wait_times = 0
        while len(self.data) < msg_size:
            self.data += self.conn.recv(4096)
            wait_times += 1
            if(wait_times > 10000): # 连接断开，重新开始accept
                print("Socket break...")
                self.data =  b''
                self.conn_state = False
                return self.conn_state, np.array(0)

        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        return self.conn_state, frame



