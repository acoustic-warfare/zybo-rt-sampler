import numpy as np
np.set_printoptions(threshold=np.inf)
import realtime_scripts.config as config
import realtime_scripts.calc_r_prime as calc_r_prime
import realtime_scripts.active_microphones as am

print('\nCalculate phase shift matrix')
c = config.PROPAGATION_SPEED
fs = int(config.fs)
N = config.N_SAMPLES
d = config.ELEMENT_DISTANCE
theta_max = config.VIEW_ANGLE/2
active_mics = am.active_microphones()   # list of active microphones

# microphone coordinates
r_prime_all, r_prime = calc_r_prime.calc_r_prime(d)
x_i = r_prime_all[0,:] # x positions of all individual microphones
y_i = r_prime_all[1,:] # y positions of all individual microphones
x_i = np.reshape(x_i, (1,len(x_i),1,1)) # reshape into 4D array
y_i = np.reshape(y_i, (1,len(y_i),1,1)) # reshape into 4D array

# scanning window
x_scan_max = config.Z*np.tan(np.deg2rad(theta_max))
x_scan_min = -x_scan_max
y_scan_max = x_scan_max/config.ASPECT_RATIO
y_scan_min = -y_scan_max

x_scan = np.linspace(x_scan_min,x_scan_max,config.MAX_RES_X)
y_scan = np.linspace(y_scan_min,y_scan_max,config.MAX_RES_Y)
x_scan = np.reshape(x_scan, (1,1,len(x_scan),1))    # reshape into 4D array
y_scan = np.reshape(y_scan, (1,1,1,len(y_scan)))    # reshape into 4D array
r_scan = np.sqrt(x_scan**2 + y_scan**2 + config.Z**2) # distance between middle of array to the xy-scanning coordinate

theta = np.arccos(config.Z/r_scan)
phi = np.arctan2(y_scan,x_scan)

f = np.linspace(0,int(fs/2),int(N/2)+1) # frequencies given by FFT
f = np.reshape(f, (len(f),1,1,1))       # reshape into 4D array
threshold_freq_lower_idx = (np.abs(f - config.threshold_freq_lower)).argmin() # lower threshold index for what frequencies to listen to
threshold_freq_upper_idx = (np.abs(f - config.threshold_freq_upper)).argmin() # upper threshold index for what frequencies to listen to

f = f[threshold_freq_lower_idx:threshold_freq_upper_idx]    # only use frequencies within lower and upper threshold range
#print(f[:,0,0,0].shape)
k = 2*np.pi*f/c      # wave number

# calculation of phase shift based on scanning window in cartesian coordinates instead of angles
phase_shift_matrix_full = -k*((x_scan*x_i + y_scan*y_i) / r_scan) # rows = frequencies, columns = array elements, depth = x_i, fourth dimension = y_i
phase_shift_full = np.exp(1j*phase_shift_matrix_full)

phase_shift = phase_shift_full[:,active_mics,:,:]
#print(np.shape(phase_shift))
