# Important

1. When you want to update a config file. Change the config.json in src folder and don't touch config.h or config.py. These will be built from the config.json when running make.
2. To run web application navigate to application and run:
```
sudo python3 manage.py runserver
```

# Known Issues
1. Sometimes a python3 process remains running in the background which leaves shared memory, semaphores and UDP port open. If this happens, run:
```
killall python3
```


# Setup

```bash
sudo apt install gcc make cython3

sudo apt-get install python3-opencv python3-pyaudio python3-matplotlib 
# python3-opencv
```