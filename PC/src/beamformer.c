/******************************************************************************
 * Title                 :   A beamformer application
 * Filename              :   src/beamformer.c
 * Author                :   Irreq
 * Origin Date           :   05/07/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 The backend for handling UDP messages and perform beamforming on the incoming
 signals.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Semaphores
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <signal.h>

#include "config.h"
#include "circular_buffer.h"
#include "udp_receiver.h"
#include "antenna/antenna.h"
#include "antenna/delay.h"

ring_buffer *rb;  // Data to be stored in
msg *client_msg;

int shmid; // Shared memory ID
int semid; // Semaphore ID

int socket_desc;

struct sembuf data_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf data_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

float ****mimo_coefficients; // TODO fix this to more healthier

float **miso_coefficients;

int whole_samples_h[MAX_RES_X * MAX_RES_Y * 1 * COLUMNS * ROWS];

pid_t pid_child;

/*
Post memory cleanup
*/
void free_coefficients()
{

    printf("Freeing memory\n");

    for (int x = 0; x < MAX_RES; x++)
    {
        for (int y = 0; y < MAX_RES; y++)
        {
            for (int i = 0; i < N_MICROPHONES; i++)
            {
                free(mimo_coefficients[x][y][i]);
            }

            free(mimo_coefficients[x][y]);
        }

        free(mimo_coefficients[x]);
    }

    free(mimo_coefficients);

    for (int i = 0; i < N_MICROPHONES; i++)
    {
        free(miso_coefficients[i]);
    }

    free(miso_coefficients);
}



/*
Remove shared memory and semafores
*/
void signal_handler()
{
    shmctl(shmid, IPC_RMID, NULL);
    semctl(semid, 0, IPC_RMID);
    close_socket(socket_desc);
    destroy_msg(client_msg);
    free_coefficients();
}

void kill_child()
{
    signal_handler();
    kill(pid_child, SIGKILL);
}

/*
Create a shared memory ring buffer
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

/*
Create the semaphore
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

/*
Steer the antenna to the direction
*/
void steer(float theta, float phi)
{
    semop(semid, &data_sem_wait, 1);
    directional_antenna_delay_coefficients(theta, // Horizontal
                                           phi, // Vertical
                                           ROWS,
                                           COLUMNS,
                                           ELEMENT_DISTANCE, // Distance between elements
                                           SAMPLE_RATE,
                                           PROPAGATION_SPEED,
                                           miso_coefficients);
    semop(semid, &data_sem_signal, 1);
}


/*
Listen in one direction (WARNING must steer before usage)
*/
void miso(float *out)
{
    float data[BUFFER_LENGTH];

    // Pin the data for retrieval
    semop(semid, &data_sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);

    // Reset output stream
    memset(out, 0, (N_SAMPLES) * sizeof(float));

    for (int i = 0; i < ROWS * COLUMNS; i++)
    {
        delay_vectorized_add(&data[i * N_SAMPLES], miso_coefficients[i], out);
    }

    for (int k = 0; k < N_SAMPLES; k++)
    {
        out[k] /= ROWS * COLUMNS - 1; // TODO Minus one to account for the faulty mic
    }
}

/*
Get power-level in multiple directions
*/
void mimo(float *image)
{
    float data[BUFFER_LENGTH];

    // Pin the data for retrieval
    semop(semid, &data_sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);

    // dummy output
    float _out[N_SAMPLES] = {0.0};
    float *out = &_out[0];

    // Start by scanning in Y-axis
    for (int y = 0; y < MAX_RES; y++)
    {
        for (int x = 0; x < MAX_RES; x++)
        {
            // Reset the output for the new direction
            memset(out, 0, (N_SAMPLES) * sizeof(float));

            for (int i = 0; i < ROWS * COLUMNS; i++)
            {

                // Delay using the EPIC delay function ;)
                delay_vectorized_add(&data[i * N_SAMPLES], mimo_coefficients[x][y][i], out);
            }

            //delay_vectorized_add(&data[2 * N_SAMPLES], mimo_coefficients[x][y][2], out);
            //delay_vectorized_add(&data[7 * N_SAMPLES], mimo_coefficients[x][y][7], out);
            //delay_vectorized_add(&data[58 * N_SAMPLES], mimo_coefficients[x][y][58], out);
            //delay_vectorized_add(&data[63 * N_SAMPLES], mimo_coefficients[x][y][63], out);

            // Compute the power-level
            float sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= (float)(ROWS * COLUMNS);
                //out[k] /= (float)4;
                //sum += powf((float)fabs((double)out[k]), 2);
                sum += powf(out[k], 2);
            }

            sum /= (float)N_SAMPLES;

            image[y * MAX_RES + x] = sum; //powf(sum, 2*MISO_POWER);
        }
    }
}

/*
TODO Fix too many nested mallocs
*/
void calculate_mimo_coefficients()
{
    mimo_coefficients = (float ****)malloc((MAX_RES) * sizeof(float ***));

    for (int x = 0; x < MAX_RES; x++)
    {
        mimo_coefficients[x] = (float ***)malloc(MAX_RES * sizeof(float **));

        for (int y = 0; y < MAX_RES; y++)
        {
            mimo_coefficients[x][y] = (float **)malloc((N_MICROPHONES) * sizeof(float *));

            for (int i = 0; i < N_MICROPHONES; i++)
            {
                mimo_coefficients[x][y][i] = (float *)malloc(N_TAPS * sizeof(float));
            }

            float horizontal = (x / (float)MAX_RES) * 2.0 * MAX_ANGLE - MAX_ANGLE;
            float vertical = (y / (float)MAX_RES) * 2.0 * MAX_ANGLE - MAX_ANGLE;

            directional_antenna_delay_coefficients(horizontal, // Horizontal
                                                   vertical,   // Vertical
                                                   ROWS,
                                                   COLUMNS,
                                                   ELEMENT_DISTANCE, // Distance between elements
                                                   SAMPLE_RATE,
                                                   PROPAGATION_SPEED,
                                                   mimo_coefficients[x][y]);
        }
    }
}

/*
Prepare coefficients
*/
void calculate_miso_coefficients()
{
    miso_coefficients = (float **)malloc((N_MICROPHONES) * sizeof(float *));

    for (int i = 0; i < N_MICROPHONES; i++)
    {
        miso_coefficients[i] = (float *)malloc(N_TAPS * sizeof(float));
    }

    directional_antenna_delay_coefficients(0.0, // Horizontal
                                           0.0,   // Vertical
                                           ROWS,
                                           COLUMNS,
                                           ELEMENT_DISTANCE, // Distance between elements
                                           SAMPLE_RATE,
                                           PROPAGATION_SPEED,
                                           miso_coefficients);
}

void myread(float *out)
{
    semop(semid, &data_sem_wait, 1);
    memcpy(out, (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);
}

void load_coefficients(int *whole_samples)
{
    memcpy(&whole_samples_h[0], whole_samples, sizeof(int) * (MAX_RES_X * MAX_RES_Y * 1 * COLUMNS * ROWS));
}

void mimo_result(int position, float *out, float *signals)
{
    int pos;
    for (int xi = 0; xi < COLUMNS; xi++)
    {
        for (int yi = 0; yi < ROWS; yi++)
        {
            // printf("%d ", whole_samples_h[position + xi * ROWS + yi]);

            pos = whole_samples_h[position + xi * ROWS + yi];

            for (int i = 0; i < N_SAMPLES - pos; i++)
            {
                out[pos + i] += signals[(xi * ROWS + yi) * N_SAMPLES + i];
            }
        }
    }
}

void work_test(float *image)
{
    float data[BUFFER_LENGTH];

    // Pin the data for retrieval
    semop(semid, &data_sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);

    // dummy output
    float _out[N_SAMPLES] = {0.0};
    float *out = &_out[0];
    int pos;

    int xi, yi;
    for (int x = 0; x < MAX_RES_Y; x++)
    {
        xi = x * MAX_RES_X * COLUMNS * ROWS;
        for (int y = 0; y < MAX_RES_X; y++)
        {
            yi = y * COLUMNS * ROWS;

            // Reset the output for the new direction
            memset(out, 0, (N_SAMPLES) * sizeof(float));

            for (int xii = 0; xii < COLUMNS; xii++)
            {
                for (int yii = 0; yii < ROWS; yii++)
                {
                    pos = whole_samples_h[xi + yi + xii * ROWS + yii];

                    for (int i = 0; i < N_SAMPLES - pos; i++)
                    {
                        out[pos + i] += data[(xii * ROWS + yii) * N_SAMPLES + i];
                    }
                }
            }

            float sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= (float)(ROWS * COLUMNS);
                sum += powf(out[k], 2);
            }

            sum /= (float)N_SAMPLES;

            image[x * MAX_RES_Y + y] = sum;
        }
    }
}

/*

Trunc-And-Sum beamformer with adaptive array configuration

*/
void mimo_truncated(float *image, int *adaptive_array, int n)
{
    float data[BUFFER_LENGTH];

    // Pin the data for retrieval
    semop(semid, &data_sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);

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
                //printf("(%d %d) ", pos_u, n);
                pos = whole_samples_h[xi + yi + s];
                for (int i = 0; i < N_SAMPLES - pos; i++)
                {
                    out[pos + i] += data[pos_u * N_SAMPLES + i];
                }
            }

            float sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= (float)n;
                sum += powf(out[k], 2);
            }

            sum /= (float)N_SAMPLES;

            image[y * MAX_RES_Y + x] = sum;
        }
    }
}

/*
Main initialization function
*/
int load(bool replay_mode)
{
    //signal(SIGINT, signal_handler);
    //signal(SIGKILL, signal_handler);
    //signal(SIGTERM, signal_handler);

    init_shared_memory();
    init_semaphore();

    calculate_mimo_coefficients();
    calculate_miso_coefficients();

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
        if(socket_desc == -1){
            return -1;
        }
        client_msg = create_msg();
        int n_arrays = receive_header_data(socket_desc);
        if (n_arrays == -1){
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