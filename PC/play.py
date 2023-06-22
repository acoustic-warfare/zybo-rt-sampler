import numpy as np
import ctypes
#import matplotlib.pyplot as plt

#import sounddevice as sd

import config 
import time

def get_antenna_data():

    lib = ctypes.cdll.LoadLibrary("./lib/libsampler.so")


    init = lib.load
    init.restype = int


    get_data = lib.myread
    get_data.restype = None
    get_data.argtypes = [
        ctypes.POINTER(ctypes.c_float)
    ]

    #get_data.argtypes = [
    #    ctypes.POINTER(ctypes.c_double)
    #]

    init()

    # N_SAMPLES = 256 * 64
    # out = np.empty(N_SAMPLES, dtype=np.float32)
    # out_pointer = out.ctypes.data_as(
    #     ctypes.POINTER(ctypes.c_float))
    
    # f = lambda arr: get_data(arr.ctypes.data_as(
    #     ctypes.POINTER(ctypes.c_float)))
    
    return get_data

f = get_antenna_data()

samples = 256 
samples = 1024

signals = np.zeros(samples * 64, dtype=np.float32)

N_SAMPLES = samples * 64
out = np.empty(N_SAMPLES, dtype=np.float32)
out_pointer = out.ctypes.data_as(
    ctypes.POINTER(ctypes.c_float))

#signals = np.zeros(samples * 64, dtype=np.float64)
#
#N_SAMPLES = samples * 64
#out = np.empty(N_SAMPLES, dtype=np.float64)
#out_pointer = out.ctypes.data_as(
#    ctypes.POINTER(ctypes.c_double))

t = (1 / 48828) * samples 
n = 100


import socket

N_MICROPHONES = 64

#D_TYPE = ctypes.c_int32
#
#class Data(ctypes.Structure):
#    _fields_ = [
#        ("arrayId", D_TYPE),  
#        ("protocolVer", D_TYPE),  # The data we care about
#        ("frequency", D_TYPE),
#        ("sampelCounter", D_TYPE),
#        ("stream", (D_TYPE*N_MICROPHONES))       
#    ]
#UDP_IP = "0.0.0.0"
#UDP_PORT = 21844
#
#"""Receive packages forever"""
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
## [Errno 98] Address already in use, https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
#sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#
#
#sock.bind((UDP_IP, UDP_PORT))
#
#print("Accepted")
#
#
#DATA_SIZE = ctypes.sizeof(Data)


import pyaudio
#p = pyaudio.PyAudio()
#stream = p.open(format=pyaudio.paFloat32,
#                channels=1,
#                rate=48828, #44100, #48828,
#                output=True) #, #,)
#                #frames_per_buffer=samples*n)

#stream = p.open(format=pyaudio.paFloat32,
#                channels=1,
#                rate=48828, #44100, #48828,
#                output=True) #, #,)
#                #frames_per_buffer=samples*n)


le = 512

#for _ in range(30):
#
#    sou = np.zeros(le, dtype=np.float32)
#    for i in range(le):
#        data = sock.recv(DATA_SIZE)
#
#        data_raw = np.frombuffer(data, dtype=D_TYPE)
#
#
#
#        #sou[i] = data_raw[4]
#
#        #print(sou[i])
#
#        #continue
#
#        #print(data_raw)
#
#        sound_ = data_raw[4:]
#
#        sound = np.array(sound_, dtype=np.float32)
#
#        # print(sound)
#
#        sound /= 2**14
#
#        # print(sound.dtype)
#
#        sou[i] = sound[0]
#
#        #print(sound[0])
#
#        # print(sound)
#
#        # print(sound[2], data_raw[3])
#
#    #stream.write(sou)
#    print(sou)
#    sd.play(sou, 48828, blocking=False)
#
##stream.stop_stream()
##stream.close()
##p.terminate()
#
#sock.close()
#exit()



p = pyaudio.PyAudio()

#rms = None
def this_callback(in_data, frame_count, time_info, status):
    # print(in_data)     # takes too long in callback
    #global rms
    #rms = audioop.rms(in_data, WIDTH)  # compute audio power
    # print(rms)  # new # takes too long in callback
    f(out_pointer)
    b = out.reshape((samples, 64))
    #time.sleep(t/1024)
    sound = b[:,4] / 1.0# / 32.0 #/ 2**15

    #print(sound.dtype)

    #print(round(abs(np.sum(sound)/sound.shape[0]), 2))

    return sound, pyaudio.paContinue
    return in_data, pyaudio.paContinue


stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=48828,
                input=False,
                output=True,
                stream_callback=this_callback,
                frames_per_buffer=samples)

stream.start_stream()

while stream.is_active():  # <--------------------------------------------
    #print(rms)    # may be losing some values if sleeping too long, didn't check
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()

exit()

for _ in range(1):
    #suu = np.zeros(samples, dtype=np.float32)
    #f(out_pointer)
    #b = out.reshape((samples, 64))
    ##time.sleep(t)
    #sound = b[:,4] / 2**14
#
    #stream.write(sound)
    #continue
    suu = np.zeros(samples*n, dtype=np.float32)
    #suu = np.zeros(samples*n, dtype=np.float64)
    for i in range(n-1):
        f(out_pointer)
        b = out.reshape((samples, 64))
        #time.sleep(t/1024)
        sound = b[:,4] / 2**14

        #sound = out[::64]

        #print(out.shape, sound.shape)

        suu[i*samples:(i+1)*samples] = sound

    #sd.play(suu, 48828, blocking=False)
    stream.write(suu)
    #stream.write(suu.astype(np.float64))


stream.stop_stream()
stream.close()
p.terminate()
