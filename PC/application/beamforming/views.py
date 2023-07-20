from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from camera import VideoCamera, gen, disc

v = VideoCamera()

def index(req):
    return render(req, "stream.html")

def stream(req):
    return StreamingHttpResponse(gen(v), content_type='multipart/x-mixed-replace; boundary=frame')

def disableBackend(req):
    v.v.value = 0
    return render(req, "stream.html")

def enableBackend(req):
    v.v.value = 1
    return render(req, "stream.html")

def disconnect(req):
    disc(v)
    return render(req, "stream.html")