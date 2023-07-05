//Do not edit this config file! Add constants and expressions in config.json and build with make. 

//General constants for both c and python.
#define N_MICROPHONES 192
#define N_SAMPLES 1024
#define N_TAPS 64
#define COLUMNS 8
#define ROWS 8
#define UDP_PORT 21844
#define UDP_IP "127.0.0.1"
#define BUFFER_LENGTH N_SAMPLES * N_MICROPHONES

//C specific constants
#define NORM_FACTOR 262144
#define KEY 1234
