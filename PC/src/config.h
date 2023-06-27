#define N_SAMPLES 128 
#define N_MICROPHONES 64
#define BUFFER_LENGTH N_SAMPLES * N_MICROPHONES
#define NORM_FACTOR 262144 //2^18 

// For convolution
#define N_TAPS 128

// For UDP-receiver
#define UDP_PORT 21844
#define SERVER_IP "10.0.0.1"

//For shared memory and semaphore in main
#define KEY 1234