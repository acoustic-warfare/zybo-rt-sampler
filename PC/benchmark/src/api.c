/******************************************************************************
 * Title                 :   A beamformer application
 * Filename              :   src/api.c
 * Author                :   Irreq, jteglund
 * Origin Date           :   20/07/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 11.3.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 This file contains code for Python (Cython) to interface with functions that share
 the same variables in the same global scope. However the necessary C functions are
 located in their respective files inside src/

 This file contains the following APIs:

 1. Connect/Disconnect - Microphone array

 2. Middle-interfaces for MIMO and MISO
    load_coefficients
    pad_mimo
    convolve_mimo_vectorized
    convolve_mimo_naive

 3. Portaudio loudspeaker playback interface
    start_playback
    stop_playback

 The interface for the beamforming algorithms and UDP-packets-receiver.

 This file spawns a child process that continiuosly stores the latest raw mic data
 to a ringbuffer for other processes or threads to access.

*/



// Semaphores
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include <unistd.h> // Error
#include <time.h>
#include <math.h>

#include "portaudio.h"

#include "config.h"
#include "api.h"
#include "receiver.h"

// Beamforming algorithms
#include "algorithms/pad_and_sum.h"
#include "algorithms/convolve_and_sum.h"
#include "algorithms/lerp_and_sum.h"

#define DEBUG 0

#define RINGBUFFER 1

/**
 * @brief MISO delay and sum beamforming - Together as one...
 *
 */

int misoshmid; // Shared memory ID
int misosemid; // Semaphore ID

int misosocket_desc;

struct sembuf misodata_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf misodata_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

pid_t misopid_child;

PaStreamParameters outputParameters, inputParameters;
PaStream *stream;

paData data;

Miso *miso;

// Global stop for audio playback
volatile sig_atomic_t stop = 0;

#if RINGBUFFER

#define BUFFER_Z N_SAMPLES * 10
typedef struct {
    float data[BUFFER_Z];
    int read_index;
    int write_index;
    int count;
} RB;

RB rb2;


// ---- BEGIN AUDIO RINGBUFFER ----
void initRingBuffer(RB *buffer)
{
    buffer->read_index = 0;
    buffer->write_index = 0;
    buffer->count = 0;
}

/**
 * @brief Write n data to a ringbuffer
 * 
 * @param buffer 
 * @param data 
 * @param n 
 */
void write_rb(RB *buffer, float *data, int n)
{
    while (buffer->count == BUFFER_Z)
    {
        usleep(1);
        ;
    }

    int i;

    for (i = 0; i < n; i++)
    {
        
        buffer->data[buffer->write_index] = data[i];
        buffer->write_index += 1;
        buffer->write_index %= BUFFER_Z;
    }

    buffer->count += n;
}

/**
 * @brief Read n data from a ringbuffer
 * 
 * @param buffer 
 * @param out 
 * @param n 
 */
void read_rb(RB *buffer, float *out, int n)
{
    while (buffer->count < n)
    {
        usleep(1);
        ;
    }

    int i;

    for (i = 0; i < n; i++)
    {
        out[i] = buffer->data[buffer->read_index];
        buffer->read_index += 1;
        buffer->read_index %= BUFFER_Z;
    }

    buffer->count -= n;
}

/**
 * @brief PortAudio callback function, should not be called from the user same as above
 *
 * @param inputBuffer
 * @param outputBuffer
 * @param framesPerBuffer
 * @param timeInfo
 * @param statusFlags
 * @param userData
 * @return int
 */
static int playback_callback(const void *inputBuffer, void *outputBuffer,
                             unsigned long framesPerBuffer,
                             const PaStreamCallbackTimeInfo *timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void *userData)
{
    RB *data = (RB *)userData;
    float *out = (float *)outputBuffer;
    (void)inputBuffer; // Cast to void to prevent compile warnings

    // Write framesPerBuffer of data into the out stream
    read_rb(data, out, framesPerBuffer);

    return paContinue;
}

#else

/* This routine will be called by the PortAudio engine when audio is needed.
** It may called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/
static int playback_callback(const void *inputBuffer, void *outputBuffer,
                             unsigned long framesPerBuffer,
                             const PaStreamCallbackTimeInfo *timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void *userData)
{
    paData *data = (paData *)userData;
    float *out = (float *)outputBuffer;
    (void)inputBuffer; // Cast to void to prevent compile warnings

    unsigned long i;

    while (!data->can_read)
    {
        usleep(1);
        ;
    }

    data->can_read = 0;
    for (i = 0; i < N_SAMPLES; i++)
    {
        *out++ = data->out[i];
    }
    return paContinue;
}

#endif






/**
 * @brief Stop the current playback audio stream
 *
 * @return int
 */
int stop_playback()
{
    printf("Stopping PortAudio Backend\n");
    PaError err;
    err = Pa_StopStream(stream);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    err = Pa_CloseStream(stream);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    Pa_Terminate();

    printf("Stopped PortAudio Backend\n");

    return err;
}

/**
 * @brief Start a Port Audio continious playback stream 
 * 
 * @param data 
 * @return int 
 */
int load_playback(paData *data)
{



    PaError err;
    err = Pa_Initialize();
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
    printf("Pa initied\n");
    outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
    if (outputParameters.device == paNoDevice)
    {
        fprintf(stderr, "Error: No default output device.\n");
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
    }
    outputParameters.channelCount = 1;         /* stereo output */
    outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    // outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultHighOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;
    printf("Starting stream\n");

#if RINGBUFFER
    initRingBuffer(&rb2);

    err = Pa_OpenStream(
        &stream,
        NULL, /* no input */
        &outputParameters,
        SAMPLE_RATE,
        N_SAMPLES * 2,
        paNoFlag, // paClipOff, /* we won't output out of range samples so don't bother clipping them */
        playback_callback,
        &rb2);
#else
    err = Pa_OpenStream(
        &stream,
        NULL, /* no input */
        &outputParameters,
        SAMPLE_RATE,
        N_SAMPLES,
        NULL, //paClipOff, /* we won't output out of range samples so don't bother clipping them */
        playback_callback,
        data);
#endif
    

    printf("Started stream\n");
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    err = Pa_StartStream(stream);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    printf("Streaming data in the background\n");

    return 0;
}

/**
 * @brief Create a stream for portaudio
 *
 */
void init_portaudio_playback()
{
    data.can_read = 0;
    for (int i = 0; i < N_SAMPLES; i++)
    {
        data.out[i] = 0.0;
    }

    load_playback(&data);
}

/**
 * @brief Create the semaphore for miso data
 *
 */
void miso_init_semaphore()
{
    misosemid = semget(KEY + 1, 1, IPC_CREAT | 0666);

    if (misosemid == -1)
    {
        perror("semget");
        exit(1);
    }

    union semun
    {
        int val;
        struct semid_ds *buf;
        short *array;
    } argument;
    argument.val = 1;

    // Set semaphore to 1
    if (semctl(misosemid, 0, SETVAL, argument) == -1)
    {
        perror("semctl");
        exit(1);
    }
}

/**
 * @brief Create shared memory for miso data, such as angle and signalbuffer
 *
 */
void miso_init_shared_memory()
{
    // Create
    misoshmid = shmget(KEY + 1, sizeof(Miso), IPC_CREAT | 0666);

    if (misoshmid == -1)
    {
        perror("shmget not working");
        exit(1);
    }

    miso = (Miso *)shmat(misoshmid, NULL, 0);

    if (miso == (Miso *)-1)
    {
        perror("shmat not working");
        exit(1);
    }

    for (int i = 0; i < N_MICROPHONES; i++)
    {
        miso->adaptive_array[i] = 0;
    }

    miso->n = 1;
    memset((void *)&miso->signals[0], 0, BUFFER_LENGTH * sizeof(float));
}

int miso_loop()
{
    steer(0);

    init_portaudio_playback();

    while (!stop)
    {

#if DEBUG
        // Time the duration of the DSP loop
        clock_t tic = clock();
#endif

        semop(misosemid, &misodata_sem_wait, 1);

        // Receive latest buffer
        get_data(&miso->signals[0]);

        // Perform MISO and write to paData
        miso_pad(&miso->signals[0], &data.out[0], &miso->adaptive_array[0], miso->n, miso->steer_offset);
        // miso_lerp(&miso->signals[0], &data.out[0], &miso->adaptive_array[0], miso->n, miso->steer_offset);
        for (int i = 0; i < N_SAMPLES; i++)
        {
            data.out[i] /= (float)miso->n;
            data.out[i] *= 20; // The amount to multiply with to get a higher volume
        }
        data.can_read = 1;

#if RINGBUFFER
        // Write the computed MISO to the audio output buffer
        write_rb(&rb2, &data.out[0], N_SAMPLES);
#endif

        semop(misosemid, &misodata_sem_signal, 1);

#if DEBUG
        clock_t toc = clock();
        printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
#endif    
    
    }

    stop_playback();

    return 0;
}

/**
 * @brief Load config for mic configuration for MISO playback.
 * 
 * Load which microphones to use as indexes in an array.
 *
 * @param adaptive_array array of microphone indexes
 * @param n length of adaptive_array
 */
void load_pa(int *adaptive_array, int n)
{
    semop(misosemid, &misodata_sem_wait, 1);
    miso->n = n;
    printf("Loading pa\n");
    for (int i = 0; i < miso->n; i++)
    {
        printf("%d ", adaptive_array[i]);
        miso->adaptive_array[i] = adaptive_array[i];
    }
    printf("\n");
    semop(misosemid, &misodata_sem_signal, 1);
}

/**
 * @brief low-level steer function that changes the steer_offset
 * of the miso coefficients pointer such that it points to the desired
 * coefficients starting-index in memory. This function should probably 
 * NOT be called on its own, but instead as a result of a function that calculates
 * the offset based on the structure of coefficients to get the desired offset
 * 
 * @param offset 
 */
void steer(int offset)
{
    semop(misosemid, &misodata_sem_wait, 1);
    miso->steer_offset = offset;
    semop(misosemid, &misodata_sem_signal, 1);
}

/**
 * @brief Must call from parent loop to stop miso playback
 *
 */
void stop_miso()
{
    // Send signal to interupt child process and stop playback
    kill(misopid_child, SIGINT);

    // Free shared memory and semaphores
    shmctl(misoshmid, IPC_RMID, NULL);
    semctl(misosemid, 0, IPC_RMID);
}

/**
 * @brief Signal handler
 *
 */
void stop_inside()
{
    stop = 1;
}

void init_miso()
{
    miso_init_shared_memory();
    miso_init_semaphore();
}

int load_miso()
{
    init_miso();
    pid_t misopid = fork(); // Fork child

    if (misopid == -1)
    {
        perror("fork");
        exit(1);
    }
    else if (misopid == 0) // Child
    {
        signal(SIGINT, stop_inside);
        miso_loop();
        exit(0); // Without exit, child returns to parent... took a while to realize
    }
    else
    {
        misopid_child = misopid;
    }

    // Return to parent
    return 0;
}

// ---- BEGIN MIMO ----

ring_buffer *rb; // Data to be stored in

msg *client_msg;

int shmid; // Shared memory ID
int semid; // Semaphore ID

int socket_desc;

struct sembuf data_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf data_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

pid_t pid_child;

/**
 * @brief Remove shared memory and semafores
 *
 */
void signal_handler()
{
    shmctl(shmid, IPC_RMID, NULL);
    semctl(semid, 0, IPC_RMID);
    close_socket(socket_desc);
    destroy_msg(client_msg);
}

/**
 * @brief Stops the receiving child process
 *
 */
void stop_receiving()
{
    signal_handler();
    kill(pid_child, SIGKILL);
}

/**
 * @brief Create a shared memory ring buffer
 *
 */
void init_shared_memory()
{
    // Create
    shmid = shmget(KEY, sizeof(ring_buffer), IPC_CREAT | 0666);

    if (shmid == -1)
    {
        perror("shmget not working");
        // strerror("shmget not working");
        exit(1);
    }

    rb = (ring_buffer *)shmat(shmid, NULL, 0);

    if (rb == (ring_buffer *)-1)
    {
        perror("shmat not working");
        // strerror("shmat not working");
        exit(1);
    }

    rb->index = 0;
    rb->counter = 0;
    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
        rb->data[i] = 0.0;
    }
}

/**
 * @brief Create the semaphore
 *
 */
void init_semaphore()
{
    semid = semget(KEY, 1, IPC_CREAT | 0666);

    if (semid == -1)
    {
        perror("semget");
        exit(1);
    }

    union semun
    {
        int val;
        struct semid_ds *buf;
        short *array;
    } argument;
    argument.val = 1;

    // Set semaphore to 1
    if (semctl(semid, 0, SETVAL, argument) == -1)
    {
        perror("semctl");
        exit(1);
    }
}

/**
 * @brief Retrieve the data located in the ring buffer
 *
 * @param out
 */
void get_data(float *out)
{
    semop(semid, &data_sem_wait, 1);
    memcpy(out, (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);
}

/**
 * @brief Main initialization function which starts
 * a child process that continiously receive data
 *
 * @param replay_mode
 * @return int
 */
int load(bool replay_mode)
{

    init_shared_memory();
    init_semaphore();

    pid_t pid = fork(); // Fork child

    if (pid == -1)
    {
        perror("fork");
        exit(1);
    }
    else if (pid == 0) // Child
    {
        // Create UDP socket:
        socket_desc = create_and_bind_socket(replay_mode);
        if (socket_desc == -1)
        {
            exit(1);
        }
        client_msg = create_msg();

        int n_arrays = receive_header_data(socket_desc);
        if (n_arrays == -1)
        {
            exit(1);
        }

        while (1)
        {
            semop(semid, &data_sem_wait, 1);

            if (receive_and_write_to_buffer(socket_desc, rb, client_msg, n_arrays) == -1)
            {
                return -1;
            }

            semop(semid, &data_sem_signal, 1);
        }

        exit(0);
    }

    pid_child = pid;

    // Return to parent
    return 0;
}


// Algorithms

/**
 * @brief Cython wrapper for MIMO using naive padding
 *
 * @param image
 * @param adaptive_array
 * @param n
 */
void pad_mimo(float *image, int *adaptive_array, int n)
{
    float signals[BUFFER_LENGTH];

    get_data(&signals[0]);

    mimo_pad(&signals[0], image, adaptive_array, n);
}




/**
 * @brief Cython wrapper for MIMO using vectorized convolve
 *
 * @param image
 * @param adaptive_array
 * @param n
 */
void convolve_mimo_vectorized(float *image, int *adaptive_array, int n)
{
    float signals[BUFFER_LENGTH];

    get_data(&signals[0]);

    mimo_convolve_vectorized(&signals[0], image, adaptive_array, n);
}

/**
 * @brief Cython wrapper for MIMO using naive convolve
 *
 * @param image
 * @param adaptive_array
 * @param n
 */
void convolve_mimo_naive(float *image, int *adaptive_array, int n)
{
    float signals[BUFFER_LENGTH];

    get_data(&signals[0]);

    mimo_convolve_naive(&signals[0], image, adaptive_array, n);
}

int whole_samples_h_[MAX_RES_X * MAX_RES_Y * ACTIVE_ARRAYS * COLUMNS * ROWS];


/**
 * @brief TODO
 * 
 * @param signals 
 * @param image 
 * @param adaptive_array 
 * @param n 
 */
void mimo_truncated_algorithm(float *signals, float *image, int *adaptive_array, int n)
{
    // dummy output
    float _out[N_SAMPLES] = {0.0};
    float *out = &_out[0];
    int pos, pos_u;

    int xi, yi;
    for (int y = 0; y < MAX_RES_Y; y++)
    {
        xi = y * MAX_RES_X * n;
        for (int x = 0; x < MAX_RES_X; x++)
        {
            yi = x * n;

            // Reset the output for the new direction
            memset(out, 0, (N_SAMPLES) * sizeof(float));

            for (int s = 0; s < n; s++)
            {
                pos_u = adaptive_array[s];
                pos = whole_samples_h_[xi + yi + s];
                for (int i = 0; i < N_SAMPLES - pos; i++)
                {
                    out[pos + i] += signals[pos_u * N_SAMPLES + i];
                }
            }

            float sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= (float)n;
                sum += powf(out[k], 2);
            }

            sum /= (float)N_SAMPLES;

            // Danger bug
            image[y * MAX_RES_X + x] = sum;
        }
    }
}


/**
 * @brief TODO
 * 
 * @param whole_samples 
 * @param n 
 */
void load_coefficients2(int *whole_samples, int n)
{
    memcpy(&whole_samples_h_[0], whole_samples, sizeof(int) * n);
}

/**
 * @brief Trunc-And-Sum beamformer with adaptive array configuration
 *
 * @param image
 * @param adaptive_array
 * @param n
 */
void mimo_truncated(float *image, int *adaptive_array, int n)
{
    float data[BUFFER_LENGTH];

    // Pin the data for retrieval
    semop(semid, &data_sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);

    mimo_truncated_algorithm(&data[0], image, adaptive_array, n);
}

/**
 * @brief Listen in a specific direction
 * 
 * @param out 
 * @param adaptive_array 
 * @param n 
 * @param steer_offset 
 */
void miso_steer_listen(float *out, int *adaptive_array, int n, int steer_offset)
{
    float signals[BUFFER_LENGTH];

    get_data(&signals[0]);

    miso_pad(&signals[0], out, adaptive_array, n, steer_offset);
}

// Local main for quick access
#if 0
int runnin = 1;
int stoper()
{
    runnin = 0;
    stop_miso();
    stop_receiving();
}

int main()
{
    int h[N_MICROPHONES] = {0};
    memset(&h[0], 0, N_MICROPHONES * sizeof(int));

    load_coefficients_pad(&h[0], N_MICROPHONES);

    load(false);
    load_miso();

    int adaptive_array[N_MICROPHONES];
    for (int i = 0; i < N_MICROPHONES; i++)
    {
        adaptive_array[i] = i;
    }

    load_pa(&adaptive_array[0], 128);
    steer(0);

    for (int i = 0; i < N_MICROPHONES; i++)
    {
        miso->adaptive_array[i] = i;
    }

    signal(SIGINT, stoper);
    signal(SIGKILL, stoper);

    while (runnin)
        usleep(100);
}

#endif