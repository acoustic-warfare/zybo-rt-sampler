import os, time

class Replay(object):
    def __init__(self, replayNumber, sleepTime=0):
        self.replay_assets = "../replays"
        self.replayCommand = "udpreplay -i lo " + self.replay_assets + "/replay" + str(replayNumber) + "/replay.pcap"
        self.sleepTime = sleepTime
    
    def beginReplayTransmission(self):
        time.sleep(self.sleepTime)
        os.system(self.replayCommand)