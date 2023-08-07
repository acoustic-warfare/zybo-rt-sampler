import numpy as np
np.set_printoptions(threshold=np.inf)
import realtime_scripts.config as config
import realtime_scripts.calc_phase_shift_cartesian as calc_phase_shift_cartesian

#n_elements = config.N_MICROPHONES
n_samples = calc_phase_shift_cartesian.N     
f = calc_phase_shift_cartesian.f      
theta = calc_phase_shift_cartesian.theta
print('Scanning window in horizontal direction:\n' + 'theta:', -int(np.rad2deg(theta)[0,0,0,round(config.MAX_RES_Y/2)]), 'to', int(np.rad2deg(theta)[0,0,config.MAX_RES_X-1,round(config.MAX_RES_Y/2)]), 'deg')

active_mics = calc_phase_shift_cartesian.active_mics
x_scan = calc_phase_shift_cartesian.x_scan
y_scan = calc_phase_shift_cartesian.y_scan
x_res = config.MAX_RES_X
y_res = config.MAX_RES_Y

phase_shift = calc_phase_shift_cartesian.phase_shift

## behöver justera med dessa för att få till bra heatmaps
threshold_heatmap = 0.2

threshold_upper = 0.8   # threshold value for detecting peak values (set between 0 and 1)
threshold_lower = 0.1   

threshold_freq_lower_idx = calc_phase_shift_cartesian.threshold_freq_lower_idx
threshold_freq_upper_idx = calc_phase_shift_cartesian.threshold_freq_upper_idx
##

def frequency_phase_shift(signal, phase_shift):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal
    FFT = FFT[threshold_freq_lower_idx:threshold_freq_upper_idx,:]
    FFT = np.reshape(FFT, (len(FFT[:,0]),len(FFT[0,:]),1,1)) # reshape into a 4D array from 2D
    FFT_shifted = FFT*phase_shift    # apply phase shift to every signal
    return FFT_shifted

def remove_neighbors(heatmap, x_index, y_index):
    n_neighbors = 5 # numbers of neighbors to remove in x- and y-direction
    x_start = x_index-n_neighbors
    x_end = x_index+n_neighbors+1
    y_start = y_index-n_neighbors
    y_end = y_index+n_neighbors+1
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0
    
    heatmap[x_start:x_end, y_start:y_end] = 0

def peak_detection(power_in,threshold_upper,threshold_lower):
    heatmap = np.zeros((config.MAX_RES_X,config.MAX_RES_Y))
    power = power_in[threshold_freq_lower_idx:,:,:]
    #for f_ind in range(threshold_freq_lower_idx, threshold_freq_upper_idx):
    for f_ind in range(0, len(power[:,0,0])):
        if (np.max(power[f_ind,:,:]) > threshold_upper*np.max(power) and  np.max(power[f_ind,:,:]) > threshold_lower):
            (x_max,y_max) = np.unravel_index(power[f_ind,:,:].argmax(), np.shape(power[f_ind,:,:])) #indexes for x- and y-position of the maximum peak
            if power[f_ind,x_max,y_max] > heatmap[x_max,y_max]:
                heatmap[x_max,y_max] = power[f_ind,x_max,y_max]
                #heatmap[x_max,y_max] += 1
            #print('Found peak value:', round(power_in[f_ind+freq_threshold_lower_idx,x_max,y_max], 8), 'at approx.', int(f[f_ind+freq_threshold_lower_idx]), 'Hz')
        if (np.max(power[f_ind,:,:]) < threshold_upper*np.max(power) and np.max(power[f_ind,:,:]) > threshold_lower):
            (x_max,y_max) = np.unravel_index(power[f_ind,:,:].argmax(), np.shape(power[f_ind,:,:])) #indexes for x- and y-position of the maximum peak

    return heatmap


def main(signal):
    signal = signal[:,active_mics]
    FFT_shifted = frequency_phase_shift(signal,phase_shift)
    FFT_power = (np.abs(np.sum(FFT_shifted,axis=1)))**2         # Power of FFT summed over all array elements


    ## använd detta blocket för standard heatmap
    heatmap = np.sum(FFT_power,axis=0)                          # power summed over all frequencies
    if np.max(heatmap) < threshold_heatmap:
        heatmap[:,:] = 0
    else:
        #print(np.max(heatmap))
        heatmap = (heatmap/np.max(heatmap))**15


    ## använd detta blocket för heatmap med peak detection
    #heatmap = peak_detection(FFT_power,threshold_upper,threshold_lower)
    #heatmap = (heatmap/np.max(heatmap))#**2

    return heatmap
