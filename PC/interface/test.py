from modules.Beamformer import Beamformer
from modules.RealtimeSoundplayer import RealtimeSoundplayer

bf = Beamformer(True)
rtsp = RealtimeSoundplayer(bf)
rtsp.play_sound()