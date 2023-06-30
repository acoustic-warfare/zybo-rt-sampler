import queue

import PC.interface.config as config

import socket
import signal
import ctypes

import numpy as np

from multiprocessing import Process, JoinableQueue, active_children
import os
import sys

# NOTE: Check if big-endian or little-endian as this is often flipped
# use:
# from ctypes import LittleEndianStructure, BigEndianStructure
# and replace Structure with LittleEndianStructure or BigEndianStructure to get the right one.
class Data(ctypes.Structure):
    _fields_ = [
        ("arrayId", config.D_TYPE),  
        ("protocolVer", config.D_TYPE),  
        ("frequency", config.D_TYPE),
        ("sampelCounter", config.D_TYPE),
        ("stream", (config.D_TYPE*config.N_MICROPHONES))  # The data we care about
    ]

DATA_SIZE = ctypes.sizeof(Data)
print(DATA_SIZE)
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


def queue_rows(q, block=False, timeout=None):
    """
     :param Queue.Queue q:
     :param bool block:
     :param int timeout:
    """
    while not q.empty():
        with read_from_q(q, block, timeout) as row:
            yield row

class _listener:

    def __init__(self, q: object):
        self._q = q

        """Receive packages forever"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # [Errno 98] Address already in use, https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.sock.bind((config.UDP_IP, config.UDP_PORT))
        self.sock.settimeout(config.TIMEOUT)

        self.running = True

        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _exit_gracefully(self, *args, **kwargs):
        """Break loop on `SIGINT` or `SIGTERM`"""
        self.running = False
        self.sock.close()
        
    def listen(self):
        """Receiving loop will put data into a queue for the algorithm to poll"""
        while self.running:
            try:
                data = self.sock.recv(DATA_SIZE)
            except TimeoutError:
                break

            data_array = np.frombuffer(data, dtype=config.D_TYPE)

            self._q.put(data_array)


        print("Done")



class AntennaReceiver:

    """Main PC - Zybo interface"""

    def __init__(self):
        self._q = JoinableQueue()

        self._listener = _listener(self._q) # 

        self.running = True

        signal.signal(signal.SIGINT, self._exit_gracefully)
        #signal.signal(signal.SIGTERM, self._exit_gracefully)
        # Starting receiver as a separate process
        self.process = Process(target=self._listener.listen)
        self.process.start()

    def _exit_gracefully(self, *args, **kwargs):
        """Break loop on `SIGINT` or `SIGTERM`"""
        self.running = False
        os.kill(self.process.pid, signal.SIGTERM)
        os._exit(0)

    def disconnect(self):
        """Wrapper for shutting down listener"""
        self._exit_gracefully()

    def get_data(self) -> np.ndarray:
        while self.running:
            counter = 0
            final_buffer = np.zeros((100, 68), dtype=config.D_TYPE)
            
            while counter < 100:
                for buffer in queue_rows(self._q):
                    
                    final_buffer[counter, :] = buffer
                    counter += 1
                    if counter >= 100:
                        break
                
            yield final_buffer

        



if __name__ == '__main__':
    receiver = AntennaReceiver()

    for i, sample in enumerate(receiver.get_data()):
        if receiver.running:
            print(sample[0])
            print(i)