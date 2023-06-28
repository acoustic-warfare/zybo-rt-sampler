# Important

1. When you want to update a config file. Change the config.json in src folder and don't touch config.h or config.py. These will be built from the config.json when running make.

# Known Issues
1. Sometimes a python3 process remains running in the background which leaves shared memory, semaphores and UDP port open. If this happens, run:
```
killall python3
```