import numpy as np
np.set_printoptions(threshold=np.inf)
import interface.config
import realtime_scripts.active_microphones_modes as amm

c = config.PROPAGATION_SPEED
fs = config.fs
N = config.N_SAMPLES
f = np.linspace(0,int(fs/2),int(N/2)+1) # frequencies after FFT
d = config.ELEMENT_DISTANCE

def calc_mode_matrices(matrix):
    mode_matrices = []
    mode_intervals = []
    active_mics_mode_list = []
    distances = c/(2*f + 0.0001)
    for mode in range(1, config.modes+1):
        if mode == 1:
            mode_idxs = (distances>0)*(distances<=(mode+1)*d)
        elif mode == config.modes:
            mode_idxs = (distances>mode*d)
        else:
            mode_idxs = (distances>mode*d)*(distances<=(mode+1)*d)

        mode_interval = np.where(mode_idxs)[0]
        #print('mode', mode, 'interval frequencies:', int(f[mode_interval[0]]), 'Hz -', int(f[mode_interval[-1]]), 'Hz')
        #print('mode', mode, 'interval indexes:', mode_interval)
        if mode == 4:       # mode 4 gives bad results. replaced by mode 3
            active_mics_mode = amm.active_microphones(mode-1)
        else:
            active_mics_mode = amm.active_microphones(mode)

        mode_matrices.append(matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:])
        mode_intervals.append(mode_interval)

        active_mics_mode_list.append(active_mics_mode)
        #print('mode', mode, 'matrix shape:', np.shape(matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:]))

    return mode_matrices, mode_intervals, active_mics_mode_list