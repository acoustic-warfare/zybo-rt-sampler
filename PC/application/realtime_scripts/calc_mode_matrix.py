import numpy as np
np.set_printoptions(threshold=np.inf)
import config
import realtime_scripts.active_microphones_modes as amm

c = config.PROPAGATION_SPEED
fs = config.fs
N = config.N_SAMPLES
f = np.linspace(0,int(fs/2),int(N/2)+1) # frequencies after FFT
d = config.ELEMENT_DISTANCE

def mode_matrix(matrix):
    mode_matrix = np.zeros_like(matrix)
    n_active_mics = np.zeros((len(f),1,1))
    #print(np.shape(n_active_mics))
    distances = c/(2*f + 0.0001)
    for mode in range(1, config.modes+1):
        if mode == 1:
            mode_idxs = (distances>0)*(distances<=(mode+1)*d)
        elif mode == config.modes:
            mode_idxs = (distances>mode*d)
        else:
            mode_idxs = (distances>mode*d)*(distances<=(mode+1)*d)

        mode_interval = np.where(mode_idxs)[0]
        #print('mode', mode, 'mode interval:', mode_interval)
        #print('mode', mode, 'interval frequencies:', int(f[mode_interval[0]]), 'Hz -', int(f[mode_interval[-1]]), 'Hz')
        #print('mode', mode, 'interval indexes:', mode_interval)
        if mode == 4:       # mode 4 gives bad results. replaced by mode 3
            active_mics_mode = amm.active_microphones(mode-1)
        else:
            active_mics_mode = amm.active_microphones(mode)
        
        mode_matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:] = matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:]
        n_active_mics[mode_interval[0]:mode_interval[-1]+1,0,0] = len(active_mics_mode)

        #print('mode', mode, 'matrix shape:', np.shape(matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:]))

    #print(n_active_mics)
    # för att testa om modesen gör som de ska
    #for i in range(len(mode_matrix[0,:,0,0])):
    #    freq = 35
    #    if mode_matrix[freq,i,0,0] != 0:
    #        print('non-zero index =', i)
    #    #else:
    #    #    print(0, 'index =', i)

    return mode_matrix, n_active_mics

#modes = 4
#for mode in modes:
#    mode_active_mics = amm.active_microphones(mode)
#    print(mode_active_mics)
#    phase_shift_mode4 = phase_shift[18:22+1,mode_active_mics,:,:]
#    print(np.shape(phase_shift_mode4))