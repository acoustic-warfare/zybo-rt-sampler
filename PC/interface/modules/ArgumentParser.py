import sys
import config

class ArgumentParser(object):
    def __init__(self):
        self.src = config.CAMERA_SOURCE
        self.replayMode = False
        self.replayNumber = ""
        if len(sys.argv) > 1 and sys.argv[1] == "replay":
            self.replayMode = True
            if len(sys.argv) == 3:
                self.src = "../replays/replay" + str(sys.argv[2]) + "/replay.avi"
                self.replayNumber = str(sys.argv[2])

    def getSrc(self):
        return self.src
    
    def getReplayMode(self):
        return self.replayMode
    
    def getReplayNumber(self):
        return self.replayNumber