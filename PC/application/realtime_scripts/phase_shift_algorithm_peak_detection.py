import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import interface.config as config
import realtime_scripts.calc_phase_shift_cartesian as calc_phase_shift_cartesian

n_elements = config.N_MICROPHONES
n_samples = config.N_SAMPLES     
f = calc_phase_shift_cartesian.f      
theta = calc_phase_shift_cartesian.theta
print('Scanning window in horizontal direction:\n' + 'theta:', -int(np.rad2deg(theta)[0,0,0,round(11/2)]), 'to', int(np.rad2deg(theta)[0,0,11-1,round(11/2)]), 'deg')

x_scan = calc_phase_shift_cartesian.x_scan
y_scan = calc_phase_shift_cartesian.y_scan
x_res = 11
y_res = 11

phase_shift = calc_phase_shift_cartesian.phase_shift
phase_shift_modes = calc_phase_shift_cartesian.phase_shift_modes
n_active_mics = calc_phase_shift_cartesian.n_active_mics

mode_matrices = calc_phase_shift_cartesian.mode_matrices
mode_intervals = calc_phase_shift_cartesian.mode_intervals
active_mics_mode_list = calc_phase_shift_cartesian.active_mics_mode_list

## behöver experimentera med dessa för att få till bra peak detection
threshold_upper = 0.8            # threshold value for detecting peak values (set between 0 and 1)
threshold_lower = 1 
#threshold_lower = 0.00017         # threshold value for detecting peak values (can be set to any value)
threshold_lower_modes = 0.00017
threshold_lower_sep = 1
freq_threshold = 300              # threshold for lowest plotted frequency
freq_threshold_idx = (np.abs(f - freq_threshold)).argmin() + 1

def frequency_phase_shift(signal, phase_shift):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal
    FFT = np.reshape(FFT, (int(n_samples/2)+1,len(FFT[0,:]),1,1)) # reshape into a 4D array from 2D
    FFT_shifted = FFT*phase_shift    # apply phase shift to every signal
    return FFT_shifted

def frequency_phase_shift_modes(signal):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal
    FFT = np.reshape(FFT, (int(n_samples/2)+1,len(FFT[0,:]),1,1)) # reshape into a 4D array from 2D
    FFT_shifted_list = []
    for mode in range(config.modes):
        #print(mode+1)
        #print('mode', mode+1, 'active mics:', len(active_mics_mode_list[mode]))
        FFT_shifted_mode = FFT[mode_intervals[mode][0]:mode_intervals[mode][-1]+1,active_mics_mode_list[mode]]*mode_matrices[mode]
        FFT_shifted_list.append(FFT_shifted_mode)
    return FFT_shifted_list

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
    #power_copy = np.copy(power)
    heatmap = np.zeros((11,11))
    power = power_in[freq_threshold_idx:,:,:]
    #for f_ind in range(freq_threshold_idx, int(n_samples/2)+1-freq_threshold_idx):
    for f_ind in range(0, len(power[:,0,0])):
        #if (np.max(power[f_ind,:,:]) > threshold_upper*np.max(power) and  np.max(power) > threshold_lower):
        if (np.max(power[f_ind,:,:]) > threshold_upper*np.max(power) and  np.max(power[f_ind,:,:]) > threshold_lower):
            (x_max,y_max) = np.unravel_index(power[f_ind,:,:].argmax(), np.shape(power[f_ind,:,:])) #indexes for x- and y-position of the maximum peak
            if power[f_ind,x_max,y_max] > heatmap[x_max,y_max]:
                heatmap[x_max,y_max] = power[f_ind,x_max,y_max]
                #heatmap[x_max,y_max] += 1
            print('Found peak value:', round(power_in[f_ind+freq_threshold_idx,x_max,y_max], 8), 'at approx.', int(f[f_ind+freq_threshold_idx]), 'Hz')
        if (np.max(power[f_ind,:,:]) < threshold_upper*np.max(power) and np.max(power[f_ind,:,:]) > threshold_lower):
            (x_max,y_max) = np.unravel_index(power[f_ind,:,:].argmax(), np.shape(power[f_ind,:,:])) #indexes for x- and y-position of the maximum peak
            #print('Not printed:', round(power[f_ind+freq_threshold_idx,x_max,y_max], 8), 'at approx.', int(f[f_ind+freq_threshold_idx]), 'Hz. Highest value:', np.max(power))

            ## Handling sources at the same frequency, doesn't work very well
            #remove_neighbors(power_copy[f_ind,:,:],x_max,y_max)
            #threshold_freq = 0.7    # set between 0 and 1
            #P = 10
            #while np.max(power_copy[f_ind,:,:])**P > threshold_freq*np.max(power)**P:
            #    (x_max,y_max) = np.unravel_index(power_copy[f_ind,:,:].argmax(), np.shape(power_copy[f_ind,:,:]))
            ##    #heatmap[x_max,y_max] += 1
            #    heatmap[x_max,y_max] += power_copy[f_ind,x_max,y_max]
            ##    #print('Found peak value:', round(FFT[f_ind,x_max,y_max]), 'at approx.', int(f[f_ind]), 'Hz')
            #    remove_neighbors(power_copy[f_ind,:,:],x_max,y_max)
    return heatmap

def peak_detection_modes(power_list,threshold_upper,threshold_lower):
    #power_copy = np.copy(power_list)
    power_stack_whole = power_list[0]
    for power_matrix in power_list[1:]:
        power_stack_whole = np.vstack((power_matrix,power_stack_whole))
    power_stack = power_stack_whole[freq_threshold_idx:,:,:]
    heatmap = np.zeros((11,11))
    #for f_ind in range(freq_threshold_idx, int(n_samples/2)+1):
    for f_ind in range(0, len(power_stack[:,0,0])):
        if (np.max(power_stack[f_ind,:,:]) > threshold_upper*np.max(power_stack) and np.max(power_stack[f_ind,:,:]) > threshold_lower):
        #if (np.max(power_stack[f_ind,:,:]) > threshold_upper*np.max(power_stack) > threshold_lower):
            (x_max,y_max) = np.unravel_index(power_stack[f_ind,:,:].argmax(), np.shape(power_stack[f_ind,:,:])) #indexes for x- and y-position of the maximum peak
            if power_stack[f_ind,x_max,y_max] > heatmap[x_max,y_max]:
                heatmap[x_max,y_max] = power_stack[f_ind,x_max,y_max]
                #heatmap[x_max,y_max] = 1
            print('Found peak value:', round(power_stack[f_ind,x_max,y_max], 8), 'at approx.', int(f[f_ind+freq_threshold_idx]), 'Hz')
    return heatmap

def calc_power(matrix, active_mics):
    matrix_power = (np.abs(np.sum(matrix,axis=1))**2)
    matrix_power = matrix_power/len(active_mics)
    return matrix_power

def main(signal):
    #### calculation of heatmap with full phase shift matrix

    ## avnänd detta blocket för standard heatmappen
    FFT_shifted = frequency_phase_shift(signal,phase_shift)
    FFT_power = (np.abs(np.sum(FFT_shifted,axis=1)))**2         # Power of FFT summed over all array elements
    heatmap = np.sum(FFT_power,axis=0)                          # power summed over all frequencies
    if np.max(heatmap) < 3000:
        heatmap[:,:] = 0
    else:
        print(np.max(heatmap))
        heatmap = (heatmap/(np.max(heatmap)+0.0001))**25


    ## avnänd detta blocket för standard heatmappen med peak detection
    #FFT_shifted = frequency_phase_shift(signal,phase_shift)
    #FFT_power = (np.abs(np.sum(FFT_shifted,axis=1)))**2         # Power of FFT summed over all array elements
    #heatmap = peak_detection(FFT_power,treshold_upper,threshold_lower)
    #heatmap = (heatmap/(np.max(heatmap)+0.0001))#**2


    #### calculation of heatmap with full mode phase shift matrix

    ## använd detta blocket för heatmap med adaptive mode configuration
    #FFT_shifted = frequency_phase_shift(signal,phase_shift_modes)
    #FFT_power = (np.abs(np.sum(FFT_shifted,axis=1))**2)         # Power of FFT summed over all array elements
    #FFT_power /= n_active_mics                                  # adaptive normalization
    #heatmap = np.sum(FFT_power,axis=0)                          # power summed over all frequencies
    #if np.max(heatmap) < 0.007:
    #    heatmap[:,:] = 0
    #else:
    #    print(np.max(heatmap))
    #    heatmap = (heatmap/(np.max(heatmap)+0.0001))**25

    ## avnänd detta blocket för heatmap med adaptive configuration och peak detection
    #FFT_shifted = frequency_phase_shift(signal,phase_shift_modes)
    #FFT_power = (np.abs(np.sum(FFT_shifted,axis=1))**2)         # Power of FFT summed over all array elements
    #FFT_power /= n_active_mics                                  # adaptive normalization
    #heatmap = peak_detection(FFT_power,threshold_upper,threshold_lower_modes)
    #heatmap = (heatmap/(np.max(heatmap)+0.0001))**7


    #### calculation with mode matrix separated

    ## använd detta blocket för heatmap med adaptive mode configuration och separerade phase shift matriser (borde gå snabbare)
   # FFT_shifted_modes = frequency_phase_shift_modes(signal)
    #power_list = list(map(calc_power, FFT_shifted_modes, active_mics_mode_list))
    #heatmap = sum(list(map(lambda x: np.sum(x,axis=0), power_list)))   # sum the power of each FFT modes over frequency, then sum all modes together
    #if np.max(heatmap) < 6:
        #heatmap[:,:] = 0
    #else:
        #print(np.max(heatmap))
        #heatmap = (heatmap/(np.max(heatmap)+0.0001))**25

    ## använd detta blocket för heatmap med peak detection av samma implementering som blockat ovan
   # FFT_shifted_modes = frequency_phase_shift_modes(signal)
    #power_list = list(map(calc_power, FFT_shifted_modes, active_mics_mode_list))
    #heatmap = peak_detection_modes(power_list,threshold_upper,threshold_lower_sep)
    #heatmap = (heatmap/(np.max(heatmap)+0.0001))**2

    return heatmap