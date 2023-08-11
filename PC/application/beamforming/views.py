from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from camera import VideoCamera, gen
import time
from multiprocessing import Process
#from play import RealtimeSoundplayer
from threading import Thread
import os  
import signal  
import sys  

v = VideoCamera()

def my_signal_handler(*args):  
    if os.environ.get('RUN_MAIN') == 'true':  
        os.system('ps aux  |  grep -i python3  |  awk \'{print $2}\'  |  xargs sudo kill -9')

signal.signal(signal.SIGINT, my_signal_handler) 

def setupThresholdSliders(req):
    sliderRange = req.GET.get('range', '5')
    sliderExp = req.GET.get('exp', '-8')
    context = {
        'sliderRange': sliderRange,
        'sliderExp': sliderExp,
    }
    threshold_str=sliderRange + 'e' + sliderExp
    threshold = float(threshold_str)

    return context, threshold

def index(req):
    global v
    v.quit()
    v = VideoCamera()
    return render(req, "stream.html")

def stream(req):
    global v
    return StreamingHttpResponse(gen(v), content_type='multipart/x-mixed-replace; boundary=frame')

def disableBackend(req):
    global v
    v.disableBeamforming()
    v.quit()
    v = VideoCamera()
    return render(req, "stream.html")

def enablePadBackend(req):
    context, threshold = setupThresholdSliders(req)
    global v
    v.quit()
    v = VideoCamera(threshold=threshold)
    v.startBeamforming(0)
    return render(req, "streamWithBackend.html", context)


def enableConvolveBackend(req):
    context, threshold = setupThresholdSliders(req)
    global v
    v.quit()
    v = VideoCamera(threshold=threshold)
    v.startBeamforming(1)
    return render(req, "streamWithBackend.html", context)

def enableThirdBackend(req):
    context, threshold = setupThresholdSliders(req)
    global v
    v.quit()
    v = VideoCamera(threshold=threshold)
    v.startBeamforming(2)
    return render(req, "streamWithBackend.html", context)

def connect(req):
    global v
    v = VideoCamera()
    return render(req, "stream.html")

def sound(req):
    global v
    v.quit()
    v = VideoCamera()
    v.startBeamforming(3)
    return render(req, "streamWithBackend.html")

def replay(req):
    # Start replay transmission
    #TODO
    # Display replay
    #TODO
    # Open stream.html
    return render(req, "stream.html")


def disconnect(req):
    global v 
    v.quit()
    return render(req, "connect.html")