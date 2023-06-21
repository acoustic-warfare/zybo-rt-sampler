import itertools
import logging
import socket
import sys
import struct
import time 
import os
from ctypes import Structure, c_byte, c_int32, sizeof
import queue


N_MICROPHONES = 64

D_TYPE = c_int32


def int_to_twos_complement_string(num):
    if num >= 0:
        binary = bin(num)[2:]  # Convert to binary string excluding the '0b' prefix
        binary = binary.zfill(32)  # Pad with leading zeros to ensure 32 bits
        binary_with_spaces = ' '.join(binary[i:i+8] for i in range(0, 32, 8))  # Add space between each byte
    else:
        # Convert to positive form by flipping the bits and adding 1
        positive_num = abs(num) - 1
        binary = bin(positive_num)[2:]  # Convert to binary string excluding the '0b' prefix
        binary = binary.zfill(32)  # Pad with leading zeros to ensure 32 bits
        
        # Flip the bits
        inverted_binary = ''.join('1' if bit == '0' else '0' for bit in binary)
        
        binary_with_spaces = ' '.join(inverted_binary[i:i+8] for i in range(0, 32, 8))  # Add space between each byte
    binary_with_spaces = binary_with_spaces[:26] + binary_with_spaces[27:]  # Remove space between last two bytes
    return binary_with_spaces

# NOTE: Check if big-endian or little-endian as this is often flipped
# use:
# from ctypes import LittleEndianStructure, BigEndianStructure
# and replace Structure with LittleEndianStructure or BigEndianStructure to get the right one.
class Data(Structure):
    _fields_ = [
        ("arrayId", D_TYPE),  
        ("protocolVer", D_TYPE),  # The data we care about
        ("frequency", D_TYPE),
        ("sampelCounter", D_TYPE),
        ("stream", (D_TYPE*N_MICROPHONES))       
    ]
UDP_IP = "0.0.0.0"
UDP_PORT = 21844

# FIFO Queue with one-time reading, meaning data is dumped when read
class read_from_q:
    def __init__(self, q, block=False, timeout=None):
        """
         :param Queue.Queue q:
         :param bool block:
         :param timeout:
        """
        self.q = q
        self.block = block
        self.timeout = timeout

    def __enter__(self):
        return self.q.get(self.block, self.timeout)

    def __exit__(self, _type, _value, _traceback):
        self.q.task_done()

# Generator function to poll the queue
def queue_rows(q, block=False, timeout=None):
    """
     :param Queue.Queue q:
     :param bool block:
     :param int timeout:
    """
    while not q.empty():
        with read_from_q(q, block, timeout) as row:
            yield row


class AntennaReceiver:
     
     def __init__(self):
          self._q = queue.Queue()

          self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



q = queue.Queue()

"""Receive packages forever"""
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# [Errno 98] Address already in use, https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

sock.bind((UDP_IP, UDP_PORT))

print("Accepted")


DATA_SIZE = sizeof(Data)

print(DATA_SIZE)

import numpy as np

data_list = []
counter_list = []

data_raw = np.empty(N_MICROPHONES, dtype=D_TYPE)
print(data_raw)
for i in range(10):
    
    data = sock.recv(DATA_SIZE)

    data_raw = np.frombuffer(data, dtype=D_TYPE)

    # Put a reference to the array in a queue for the transmitter
                    self._receive_q.put(cp_arr)



    print(data_raw)



    data_list.append(data)

#for data in data_list:
#
#    d = Data.from_buffer_copy(data)
#
#    microphones = []
#    counter_list.append(d.sampelCounter)
#
#    # Printing each mic data as integers
#    for i in range(1, 65):
#        field_name = f"mic_{i}"
#        mic_data = int_to_twos_complement_string(getattr(d, field_name))
#        microphones.append(mic_data)
#
#first = counter_list[0]
#
#
#
#arr = np.array(counter_list)
#
#arr -= first
#
#import matplotlib.pyplot as plt
#
#plt.plot(arr)
#plt.show()