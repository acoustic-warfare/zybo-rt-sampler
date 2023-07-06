//Do not edit this config file! Add constants and expressions in config.json and build with make. 

//General constants for both c and python.
#define N_MICROPHONES 192
#define N_SAMPLES 1024
#define N_TAPS 64
#define COLUMNS 8
#define ROWS 8
#define MAX_RES 20
#define MAX_ANGLE 45.0
#define UDP_PORT 21844
#define SAMPLE_RATE 48828.0
#define ELEMENT_DISTANCE 0.02
#define PROPAGATION_SPEED 340.0
#define UDP_IP "10.0.0.1"
#define UDP_REPLAY_IP "127.0.0.1"
#define BUFFER_LENGTH N_SAMPLES * N_MICROPHONES

//C specific constants
#define NORM_FACTOR 262144
#define MISO_POWER 5
#define KEY 1234
