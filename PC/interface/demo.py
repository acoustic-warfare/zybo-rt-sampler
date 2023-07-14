#import time
import matplotlib.pyplot as plt

import numpy as np

import config
#from lib.antenna import connect, disconnect, receive

#from lib import antenna

from lib.microphone_array import connect, disconnect, receive

connect(True)

data = np.zeros((config.N_MICROPHONES, config.N_SAMPLES), dtype=np.float32)
#time.sleep(2)

out = np.zeros(config.N_SAMPLES*10, dtype=np.float32)

for i in range(10):
    receive(data)
    out[i*config.N_SAMPLES:(i+1)*config.N_SAMPLES] = data[0]
    
plt.plot(out)
plt.show()

disconnect()

exit()