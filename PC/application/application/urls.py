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
from beamforming.views import index, stream, disableBackend, enablePadBackend, disconnect, connect, enableConvolveBackend, replay, sound, enableThirdBackend

def fun(s):
    hej = 0
    return hej
urlpatterns = [
    path('', index),
    path('disable/', disableBackend),
    path('enableBackend1/', enablePadBackend),
    path('enableBackend2/', enableConvolveBackend),
    path('disconnect/', disconnect),
    path('connect/', connect),
    path('monitor/', stream),
    path('sound/', sound),
    path('replay/', replay),
    path('enableBackend3/', enableThirdBackend)
]
