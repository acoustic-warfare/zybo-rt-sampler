from modules.RealtimeSoundplayer import RealtimeSoundplayer
from modules.Beamformer import Beamformer

bf = Beamformer()
rtsp = RealtimeSoundplayer(bf)
rtsp.get_samples()