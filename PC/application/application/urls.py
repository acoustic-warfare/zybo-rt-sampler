"""application URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.http import StreamingHttpResponse, HttpResponse
from beamforming.views import index

from camera import VideoCamera, gen

def fun(s):
    hej = HttpResponse("<html><head><title>Video Streaming Demonstration</title></head><body><button type=\"button\" onclick=\"alert('Hello world!')\">Trunc and sum backend</button><button type=\"button\" onclick=\"alert('Hello world!')\">Convolve</button><img src=\"http://127.0.0.1:8000/monitor\"></body></html>")
    return hej
urlpatterns = [
    path('', index),
    path('admin/', admin.site.urls),
    path('monitor/', lambda r: StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')),
]
