from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from camera import VideoCamera, gen
import time

v = VideoCamera()

def index(req):
    return render(req, "stream.html")

def stream(req):
    return StreamingHttpResponse(gen(v), content_type='multipart/x-mixed-replace; boundary=frame')

def disableBackend(req):
    global v
    v.v.value = 0
    return render(req, "stream.html")

def enableBackend(req):
    global v
    v.v.value = 1
    v.startBeamforming()
    return render(req, "stream.html")

def connect(req):
    global v
    v = VideoCamera()
    return render(req, "stream.html")

def disconnect(req):
    global v 
    v.quit()
    return render(req, "connect.html")