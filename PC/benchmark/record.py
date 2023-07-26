from lib.beamformer import *

from interface import config

import numpy as np
import time

import datetime

ITERATIONS = 50

if __name__ == "__main__":
    samples = ITERATIONS*config.N_SAMPLES
    seconds = samples/config.SAMPLE_RATE
    print(f"Will record for {seconds} seconds")

    now=datetime.datetime.now()

    file_name = now.isoformat() + f" {config.SAMPLE_RATE}Hz {seconds}s [{samples}]" + ".npy"
    file_name = file_name.replace(" ", "_")
    print(file_name)

    total_data = np.zeros((config.N_MICROPHONES, ITERATIONS*config.N_SAMPLES), dtype=np.float32)

    data = np.zeros((config.N_MICROPHONES, config.N_SAMPLES), dtype=np.float32)

    connect()
    try:
        for i in range(ITERATIONS):
            receive(data)
            total_data[:,i*config.N_SAMPLES:(i+1)*config.N_SAMPLES] = data.copy()

    finally:
        np.save(file_name, total_data)
        disconnect()
