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

#include "config.h"
#include "receiver.h"

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
        exit(1);
    }

    rb = (ring_buffer *)shmat(shmid, NULL, 0);

    if (rb == (ring_buffer *)-1)
    {
        perror("shmat not working");
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
            return -1;
        }
        client_msg = create_msg();
        int n_arrays = receive_header_data(socket_desc);
        if (n_arrays == -1)
        {
            return -1;
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
    }

    pid_child = pid;

    // Return to parent
    return 0;
}



// Algorithms

#include "algorithms/pad_and_sum.h"

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


#include "algorithms/convolve_and_sum.h"

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
#include <math.h>

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
                // printf("(%d %d) ", pos_u, n);
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