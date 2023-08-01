import config
import numpy as np

def active_microphones(mode):
    # depending on the chosen mode, the correct microphone indexes are calculated
    # and stored in the list active_mics
    #       mode = 1: alla mikrofoner, 
    #       mode = 2: varannan
    #       mode = 3: var tredje
    #       mode = 4: var fjärde
    #       (visualisera array setup med att sätta plot_setup = 1 i config.py)
     
    rows = np.arange(0, config.rows, mode)                              # number of rows in array
    columns = np.arange(0, config.columns*config.ACTIVE_ARRAYS, mode)   # number of columns in array

    mics = np.linspace(0, config.N_MICROPHONES-1, config.N_MICROPHONES)           # vector holding all microphone indexes for all active arrays
    arr_elem = config.rows*config.columns                               # number of elements in one array

    # microphone indexes for one array, in a matrix
    microphones = np.linspace(0, config.rows*config.columns-1,config.rows*config.columns).reshape((config.rows, config.columns))

    # for each additional array, stack a matrix of the microphone indexes of that array
    for a in range(config.ACTIVE_ARRAYS-1):
        a += 1
        array = mics[0+a*arr_elem : arr_elem+a*arr_elem].reshape((config.rows, config.columns))
        microphones = np.hstack((microphones, array))

    # take out the active microphones from the microphones matrix, save in list active_mics
    active_mics = []
    for r in rows:
        for c in columns:
            mic = microphones[r,c]
            active_mics.append(int(mic))

    # sort the list such that the mic indexes are in ascending order
    active_mics = np.sort(active_mics)
    return active_mics