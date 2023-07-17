import sys
import config
import argparse

class ArgParser(object):
    def __init__(self):
        self.src = config.CAMERA_SOURCE
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--convolution", help="changes the beamforming algorithm to convolution instead of pad and sum", action="store_true") 
        self.parser.add_argument("--replay", help="changes playback mode to replay mode")
        self.args = self.parser.parse_args()
        self.replayMode = False
        self.replayNumber = ""

        
        if self.args.replay:
            self.replayMode = True
            self.replayNumber = self.args.replay
            self.src = "../replays/replay" + self.args.replay + "/replay.avi"
    
    def getSrc(self):
        return self.src
    
    def getReplayMode(self):
        return self.replayMode
    
    def getReplayNumber(self):
        return self.replayNumber
    
    def getBeamformingAlgorithm(self):
        return self.args.convolution
