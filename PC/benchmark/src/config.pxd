# External Configs initially defined in src/config.json

# Usage: from config cimport *

cdef extern from "config.h":
    int N_MICROPHONES
    int N_SAMPLES
    int N_TAPS
    int COLUMNS
    int ROWS
    int MAX_RES
    int MAX_RES_X
    int MAX_RES_Y
    float MAX_ANGLE
    float SAMPLE_RATE
    float ELEMENT_DISTANCE
    float ARRAY_SEPARATION
    int ACTIVE_ARRAYS
    int SKIP_N_MICS
    float PROPAGATION_SPEED
    int MISO_POWER
    float VIEW_ANGLE
    int APPLICATION_WINDOW_WIDTH
    int APPLICATION_WINDOW_HEIGHT
    int CAMERA_SOURCE
    int FLIP_IMAGE