from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from camera import VideoCamera, gen
import time
from multiprocessing import Process
from play import RealtimeSoundplayer
from threading import Thread
import os  
import signal  
import sys  

v = VideoCamera()

def my_signal_handler(*args):  
    if os.environ.get('RUN_MAIN') == 'true':  
        os.system('ps aux  |  grep -i python3  |  awk \'{print $2}\'  |  xargs sudo kill -9')

signal.signal(signal.SIGINT, my_signal_handler) 


def index(req):
    threshold = req.GET.get('t', '-5')
    print(threshold)
    context = {
        't': threshold
    }
    global v
    v.quit()
    v = VideoCamera()
    return render(req, "stream.html", context)

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
    global v
    v.quit()
    v = VideoCamera()
    v.startBeamforming(0)
    return render(req, "stream.html")


def enableConvolveBackend(req):
    global v
    v.quit()
    v = VideoCamera()
    v.startBeamforming(1)
    return render(req, "stream.html")

def enableThirdBackend(req):
    global v
    v.quit()
    v = VideoCamera()
    v.startBeamforming(2)
    return render(req, "stream.html")

def connect(req):
    global v
    v = VideoCamera()
    return render(req, "stream.html")

def replaySelection(req):
    # If connected, release video capture object
    #TODO
    # Disconnect and reconnect with replay mode
    #TODO
    # Open stream.html
    return render(req, "replay_selection.html")

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